"""Golden characterization for the per-env BackendFamily declarations.
Pins that every env declares the family the audit verified, and that the only
derived warm strategy is SINGLETON -> EAGER.

Imports the real env main modules (register_family runs at import); reads the real
registry. No docker / no env-server.

Run: uv run pytest tests/gym/services/test_services_family_declarations.py -q
"""

from __future__ import annotations

import sys

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

registry = sys.modules["lite.gym.registry"]

# The envs and their audit-verified family.
EXPECTED: dict[str, BackendFamily] = {
    "androidworld": BackendFamily.DEDICATED,
    "androidlab": BackendFamily.DEDICATED,
    "lite.osworld": BackendFamily.DEDICATED,
    "lite.cuagym": BackendFamily.DEDICATED,
    "lite.scalecua": BackendFamily.DEDICATED,
    "lite.cuaworld.pymol": BackendFamily.DEDICATED,
    "osworld": BackendFamily.DEDICATED,  # official OSWorld: VM-in-Docker
    "osworld_2": BackendFamily.REMOTE,  # release-matched Worker owns the VM
    "mobileworld": BackendFamily.DEDICATED,
    "captcha": BackendFamily.DEDICATED,
    "lite.demo": BackendFamily.DEDICATED,
    "waa": BackendFamily.DEDICATED,
    "webgym": BackendFamily.SINGLETON,
    "mobilegym": BackendFamily.SINGLETON,
    "browsergym": BackendFamily.SINGLETON,
    "online_mind2web": BackendFamily.SINGLETON,
    "webharbor.webvoyager": BackendFamily.SINGLETON,
    "osworld_g": BackendFamily.PURE,
    "screenspot_pro": BackendFamily.PURE,
}


@pytest.mark.parametrize("env_id,family", list(EXPECTED.items()))
def test_declared_family_matches_audit(env_id, family):
    import_env_main_or_skip(env_id)
    assert family_of(env_id) is family


def test_all_declared_coverage_floor():
    """Every IMPORTABLE env declares a family, and at least 10 of the 18 must
    be importable on ANY host — FAILS (not skips) below the floor: silent
    zero-coverage is the failure mode the per-env skips must not reintroduce."""
    importable = importable_env_ids(EXPECTED)
    for env_id in importable:
        assert env_id in registry._families, f"{env_id} imported but declared no family"
    assert len(importable) >= 10, f"only {len(importable)}/18 envs importable (vacuous coverage)"


def test_all_cuaworld_leaves_have_family_and_services():
    # own import — do NOT rely on a sibling item on the same xdist worker
    __import__("lite.gym.envs.lite.cuaworld.main")
    leaves = _production_cuaworld_leaves()
    assert len(leaves) == 40
    assert set(leaves) <= registry._declared_env_ids
    for env_id in leaves:
        assert family_of(env_id) is BackendFamily.DEDICATED
        assert isinstance(services_for(env_id), EnvServices)
        assert warm_strategy(env_id) is WarmStrategy.NONE


@pytest.mark.parametrize("env_id", list(EXPECTED))
def test_warm_pool_strategy_retired_for_all_non_singletons(env_id):
    """Only SINGLETON envs derive a warm strategy; everything else is NONE."""
    import_env_main_or_skip(env_id)
    if EXPECTED[env_id] is BackendFamily.SINGLETON:
        assert warm_strategy(env_id) is WarmStrategy.EAGER
    else:
        assert warm_strategy(env_id) is WarmStrategy.NONE


SINGLETONS = [e for e, f in EXPECTED.items() if f is BackendFamily.SINGLETON]


@pytest.mark.parametrize("env_id", SINGLETONS)
def test_singleton_overrides_ensure_and_health(env_id):
    """Structural contract: a SINGLETON must override ensure + health
    (so `available` reflects real readiness). Enforced as a test, not a runtime
    assert, so a future SINGLETON missing health() is caught here (this also pins
    that browsergym's new health() landed)."""
    import_env_main_or_skip(env_id)
    svc = services_for(env_id)
    assert isinstance(svc, EnvServices), f"{env_id} SINGLETON must register EnvServices"
    assert type(svc).ensure is not EnvServices.ensure, f"{env_id} must override ensure()"
    assert type(svc).health is not EnvServices.health, f"{env_id} must override health()"


@pytest.mark.parametrize("env_id,family", list(EXPECTED.items()))
def test_derivations_consistent(env_id, family):
    import_env_main_or_skip(env_id)
    # warm strategy
    if family is BackendFamily.SINGLETON:
        assert warm_strategy(env_id) is WarmStrategy.EAGER
    else:
        assert warm_strategy(env_id) is WarmStrategy.NONE
    # reconcile mode
    assert (
        reconcile_mode(family)
        is {
            BackendFamily.DEDICATED: ReconcileMode.PER_INSTANCE,
            BackendFamily.SINGLETON: ReconcileMode.BOOT_ONLY,
            BackendFamily.PURE: ReconcileMode.NONE,
            BackendFamily.REMOTE: ReconcileMode.NONE,
        }[family]
    )
    # has-lifecycle
    assert has_lifecycle(family) is (family in (BackendFamily.DEDICATED, BackendFamily.SINGLETON))


# ── section B1/section B2 finalizer (validate_declarations) ────────────────────────────────


class _FakeSvc(EnvServices):
    def register_tasks(self, env_id):
        pass


@pytest.fixture()
def _fake_env():
    """Register a throwaway env id; clean every table afterwards."""
    eid = "zz_fake_finalizer_env"
    yield eid
    registry._services.pop(eid, None)
    registry._families.pop(eid, None)


def test_finalizer_services_without_family_raises(_fake_env):
    from lite.gym.services import validate_declarations

    registry._services[_fake_env] = _FakeSvc()
    with pytest.raises(RuntimeError, match="no BackendFamily"):
        validate_declarations(_fake_env)


def test_finalizer_lifecycle_family_without_services_raises(_fake_env):
    from lite.gym.services import validate_declarations

    registry._families[_fake_env] = BackendFamily.SINGLETON
    with pytest.raises(RuntimeError, match="no EnvServices"):
        validate_declarations(_fake_env)


def test_finalizer_dedicated_without_live_ids_override_raises(_fake_env):
    """section B1: EnvServerResource-style DEDICATED env whose services keep the base
    live_ids (None = 'no per-instance world') would silently disable
    ghost/orphan detection — must fail loud."""
    from lite.gym.services import validate_declarations

    registry._families[_fake_env] = BackendFamily.DEDICATED
    registry._services[_fake_env] = _FakeSvc()  # no live_ids override
    with pytest.raises(RuntimeError, match="live_ids"):
        validate_declarations(_fake_env)


def test_finalizer_ok_after_pool_retirement(_fake_env, caplog):
    """A DEDICATED env with a live_ids-overriding services passes clean."""
    import logging as _logging

    from lite.gym.services import validate_declarations

    class _ReconcilingSvc(_FakeSvc):
        def live_ids(self, env_id, scope):
            return set()

    registry._families[_fake_env] = BackendFamily.DEDICATED
    registry._services[_fake_env] = _ReconcilingSvc()
    with caplog.at_level(_logging.WARNING, logger="lite.gym.services"):
        validate_declarations(_fake_env)
    assert not caplog.records


def test_finalizer_passes_for_every_real_env():
    """The finalizer must be clean on the ENTIRE real env tree (both modes run
    it; a false positive here would break direct-mode imports)."""
    from lite.gym.services import validate_declarations

    for env_id in EXPECTED:
        validate_declarations(env_id)


# ── finalizer WIRING: _import_env itself must fire validate_declarations ─────
# The tests above call validate_declarations directly; these prove the
# direct-import path is actually wired to it (both _import_env call sites:
# leaf main.py import and dotted umbrella-sub-env), so a mis-declared env
# fails at first make/task-probe — not silently at reap time.


def test_finalizer_fires_from_direct_import_leaf(monkeypatch, tmp_path):
    """Leaf call site: envs/<name>/main.py exists, its import registers
    services WITHOUT a family → _import_env must re-raise the clause-(a)
    RuntimeError (and must not poison the imported-cache)."""
    eid = "zz_fake_leaf_env"
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
    # A real main.py file (so the leaf branch is taken) whose "import" is a
    # pre-seeded sys.modules entry — importlib returns it without executing
    # anything, keeping the test hermetic. The services-without-family state
    # stands in for what the module's import would have registered.
    (tmp_path / eid).mkdir()
    (tmp_path / eid / "main.py").write_text("")
    monkeypatch.setattr(registry, "_ENVS_DIR", tmp_path)
    import types

    mod_name = f"lite.gym.envs.{eid}.main"
    sys.modules[mod_name] = types.ModuleType(mod_name)
    registry._services[eid] = _FakeSvc()
    try:
        with pytest.raises(RuntimeError, match="no BackendFamily"):
            registry._import_env(eid)
        assert eid not in registry._imported, (
            "a failed finalizer must not cache the import (retry must re-raise)"
        )
    finally:
        sys.modules.pop(mod_name, None)
        registry._services.pop(eid, None)
        registry._imported.pop(eid, None)


def test_finalizer_fires_from_direct_import_dotted(monkeypatch):
    """Dotted call site: a sub-env with no dir of its own resolves via the
    umbrella parent; the finalizer runs on the DOTTED name before caching."""
    eid = "zz_fake_umbrella.sub"
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
    registry._services[eid] = _FakeSvc()
    try:
        with pytest.raises(RuntimeError, match="no BackendFamily"):
            registry._import_env(eid)
        assert eid not in registry._imported
    finally:
        registry._services.pop(eid, None)
        registry._imported.pop(eid, None)
        registry._imported.pop("zz_fake_umbrella", None)


def test_cuaworld_pymol_importable_not_silently_skipped():
    """A conftest ``_env_main_module`` resolver revert degrades the
    ``lite.cuaworld.*`` rows to named SKIPs (ModuleNotFoundError) that the
    coverage floors are too coarse to notice — this must-import assert makes
    the revert loud (the U17 pattern from test_metadata_invariant)."""
    assert "lite.cuaworld.pymol" in importable_env_ids(EXPECTED)
