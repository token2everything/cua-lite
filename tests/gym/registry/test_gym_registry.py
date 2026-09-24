"""Regression tests for ``lite.gym.registry`` itself.

Locks invariants that silently broke during the env_id naming refactor
(``lite-osworld`` → ``lite.osworld``). Without these,
the registry can fail to import nested env modules and ``env_ids()`` can
hide leaf envs behind their umbrella namespace.

Run:
    uv run pytest tests/gym/registry/test_gym_registry.py -v
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

import lite.gym as gym
from lite.gym.registry import (
    _clear_env_registration,
    _env_make_kwargs,
    _env_supported_kwargs,
    _has_env_registration,
    _import_all,
    _import_env,
    _import_locks,
    _imported,
    _imported_remote_urls,
    _services_started,
    _specs,
    _splits,
    _tasks_registered,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _direct_mode_by_default(monkeypatch):
    """Registry tests opt into remote routing explicitly when they need it."""
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_TOKEN", raising=False)


def _reset_registry_state(env_id: str) -> None:
    """Remove cached imports/splits for one env_id so the test starts clean."""
    _imported.pop(env_id, None)
    _imported_remote_urls.pop(env_id, None)
    _splits.pop(env_id, None)
    for key in [k for k in _specs if k.startswith(f"{env_id}@")]:
        _specs.pop(key, None)
    _env_make_kwargs.pop(env_id, None)
    _env_supported_kwargs.pop(env_id, None)
    _import_locks.pop(env_id, None)
    # Evict the registry-level lazy-registration cache for this env. Without
    # this, a subsequent ensure_registered(env_id)/task_ids(env_id) on the same
    # xdist worker short-circuits (registry.py: ``if env_id in _tasks_registered:
    # return``) and is a no-op — leaving ``_splits`` empty and masking a real
    # registration regression. (Audit #10.)
    _tasks_registered.discard(env_id)
    _services_started.discard(env_id)
    # Evict the env's module too. importlib caches modules, so without this a
    # re-``_import_env`` is a no-op (module-level registration never re-runs)
    # whenever another test on the same xdist worker already imported it —
    # leaving ``_splits`` empty and flaking the dotted-namespace assertions.
    # ``env_id`` dots map to the package path: lite.osworld →
    # ``lite.gym.envs.lite.osworld.main``.
    mod_name = f"lite.gym.envs.{env_id}.main"
    # webgym/browsergym keep their OWN module-level ``_tasks_registered`` bool
    # guarding register_tasks(); clear it on the live module before evicting so a
    # not-yet-evicted reference (or a re-import that finds the module cached) does
    # not stay stuck in the "already registered" state.
    mod = sys.modules.get(mod_name)
    if mod is not None and hasattr(mod, "_tasks_registered"):
        mod._tasks_registered = False
    sys.modules.pop(mod_name, None)


def test_import_env_handles_dotted_namespace():
    """``_import_env('lite.osworld')`` must walk ``envs/lite/osworld/main.py``
    and trigger the env's registrations. Pre-refactor the path check used
    ``_ENVS_DIR / 'lite.osworld'`` (a literal directory that doesn't exist),
    causing silent no-op.
    """
    _reset_registry_state("lite.osworld")
    _import_env("lite.osworld")
    assert "lite.osworld" in _imported, (
        "_import_env('lite.osworld') silently failed — likely the dotted-name "
        "path resolution broke. Check registry._import_env."
    )
    assert "lite.osworld" in _splits, (
        "lite.osworld imported but no tasks registered. Check that "
        "lite/gym/envs/lite/osworld/main.py registers tasks under env_id 'lite.osworld'."
    )


def test_import_env_handles_dotted_via_umbrella_parent():
    """When an env directory has no ``main.py`` of its own (or only an
    umbrella), ``_import_env`` should fall back to the parent namespace.
    Currently every leaf has its own main.py so this is mostly a regression
    canary — it should not raise.
    """
    # Use a name that doesn't exist as a directory, with a real parent.
    # The fallback should walk up to "lite" and import it (idempotent).
    _import_env("lite")
    assert "lite" in _imported


def test_partial_lite_child_import_still_imports_lite_umbrella():
    """A partially populated ``lite.*`` registry must not satisfy ``_import_env('lite')``.

    Importing one child module directly can register e.g. ``lite.osworld``
    before the ``lite`` umbrella has imported siblings such as ``lite.cuaworld``.
    ``registered_env_ids()`` must still import the umbrella and expose makeable
    cuaworld leaves instead of the bare namespace.
    """
    code = """
import json
import os

os.environ.pop("CUA_LITE_ENV_SERVER_URL", None)
os.environ.pop("CUA_LITE_ENV_SERVER_TOKEN", None)

import lite.gym.envs.lite.osworld.main  # noqa: F401
from lite.gym.registry import registry

ids = registry.registered_env_ids()
print(json.dumps({
    "has_namespace": "lite.cuaworld" in ids,
    "has_pymol": "lite.cuaworld.pymol" in ids,
    "cuaworld_count": sum(i.startswith("lite.cuaworld.") for i in ids),
}))
"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("CUA_LITE_ENV_SERVER_URL", None)
    env.pop("CUA_LITE_ENV_SERVER_TOKEN", None)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout.splitlines()[-1])
    assert got["has_namespace"] is False
    assert got["has_pymol"] is True
    assert got["cuaworld_count"] >= 1


def test_env_ids_returns_leaf_envs():
    """``env_ids()`` must return actual leaf env_ids (e.g. ``lite.osworld``)
    rather than just their umbrella namespace (``lite``). Pre-refactor the
    function returned only top-level dirs, hiding ``lite.osworld`` and
    ``lite.demo`` behind the umbrella ``lite`` directory.
    """
    ids = gym.registry.env_ids()
    assert "lite.osworld" in ids, f"lite.osworld missing from env_ids: {ids}"
    assert "lite.demo" in ids, f"lite.demo missing from env_ids: {ids}"
    # Umbrella name itself should NOT be a leaf when sub-envs exist.
    assert "lite" not in ids, (
        f"Umbrella 'lite' leaked into env_ids — should be hidden when "
        f"sub-envs (lite.osworld, lite.demo) are surfaced as leaves: {ids}"
    )


def test_dotted_env_keys_register_and_resolve():
    """End-to-end: tasks registered under dotted env_ids must be reachable
    through ``_specs`` lookup. Catches the case where ``register()``
    succeeds but the registry indexes them under an obsolete env_id form.
    """
    # Warm registry by querying tasks (triggers _import_env).
    eval_tasks = gym.registry.task_ids("lite.osworld", split="eval")
    assert len(eval_tasks) > 0, (
        f"lite.osworld has zero eval tasks — registration may have silently "
        f"failed. _splits['lite.osworld'] = {_splits.get('lite.osworld')}"
    )

    # Pick one and confirm a corresponding env_key exists in _specs.
    sample_task = eval_tasks[0]
    sample_key = f"lite.osworld@{sample_task}"
    assert sample_key in _specs, (
        f"Registered task {sample_task!r} not findable as env_key {sample_key!r} "
        f"in _specs. Possible env_key/separator mismatch."
    )

    # Spec must round-trip env_id and task_id correctly through metadata.others.
    spec = _specs[sample_key]
    assert spec.key == sample_key
    assert spec.metadata is not None
    assert spec.metadata.others["task_id"] == sample_task


def test_register_accepts_generic_metadata_stub():
    from lite.core import LiteGenericMetadata
    from lite.gym.registry import register, registry

    env_id = "faketest_generic_metadata"
    key = f"{env_id}@t1"
    _reset_registry_state(env_id)

    try:
        register(
            key,
            entry_point=_carry_stub_entry_point,
            split="eval",
            metadata=LiteGenericMetadata(others={"source": "unit"}),
        )

        meta = registry.task_metadata(env_id, "t1")
        assert isinstance(meta, LiteGenericMetadata)
        assert meta.dims == ()
        assert meta.others == {
            "source": "unit",
            "task_id": "t1",
            "env_id": env_id,
        }
        assert meta.to_dict() == {
            "metadata_kind": "generic",
            "dims": [],
            "extra_tool_schemas": [],
            "others": {
                "source": "unit",
                "task_id": "t1",
                "env_id": env_id,
            },
        }
    finally:
        _reset_registry_state(env_id)


def test_external_registration_module_replays_after_registry_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached out-of-tree registration modules must replay after registry clear."""
    env_id = "faketest_external_cached_file"
    module_name = "external_cached_registration"
    key = f"{env_id}@t1"
    (tmp_path / f"{module_name}.py").write_text(
        "\n".join([
            "from lite.gym.registry import register",
            "IMPORT_COUNT = globals().get('IMPORT_COUNT', 0) + 1",
            "",
            "def _entry_point(**_kwargs):",
            "    raise RuntimeError('not constructed')",
            "",
            f"register({key!r}, entry_point=_entry_point, split='eval')",
        ])
    )

    _reset_registry_state(env_id)
    sys.modules.pop(module_name, None)
    importlib.invalidate_caches()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("CUA_LITE_REGISTRATION_MODULES", module_name)
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)

    try:
        _import_env(env_id)
        module = sys.modules[module_name]
        assert module.IMPORT_COUNT == 1
        assert key in _specs

        _clear_env_registration(env_id)
        assert not _has_env_registration(env_id)
        assert module_name in sys.modules

        _import_env(env_id)
        module = sys.modules[module_name]
        assert module.IMPORT_COUNT == 2
        assert key in _specs
        assert _splits[env_id] == {"eval": ["t1"]}
        assert _imported[env_id] == "local"
    finally:
        _reset_registry_state(env_id)
        sys.modules.pop(module_name, None)


def test_lite_demo_registers_under_dotted_env_id():
    """Mirror of test_dotted_env_keys_register_and_resolve for lite.demo —
    catches the case where one sub-env breaks but the other works."""
    tasks = gym.registry.task_ids("lite.demo")
    # task_ids() with no split returns dict {split: [task_ids]}
    assert isinstance(tasks, dict)
    total = sum(len(ts) for ts in tasks.values())
    assert total > 0, f"lite.demo has zero tasks across all splits: {tasks}"


def test_browsergym_sub_envs_register_under_dotted_form():
    """``browsergym.{miniwob,webarena,visualwebarena}`` are
    PROGRAMMATICALLY registered (no per-sub-env directory). They are NOT
    listed by ``env_ids()`` (which is a fast directory scan) but MUST be
    reachable via ``task_ids()`` and have keys in ``_splits``.

    Catches the regression where the dotted form fails to register either
    because the umbrella main.py isn't imported, or because the env_id
    rename to dotted didn't take effect uniformly.
    """
    # Registration is lazy: importing the umbrella no longer populates _splits —
    # the catalog is realised by browsergym's register_tasks() hook, fired here
    # via ensure_registered() (the same register-only path task_ids() takes,
    # WITHOUT booting any backend). Skip if the umbrella's deps aren't installed
    # locally — the dotted-form invariant can't be checked without registration
    # firing.
    from lite.gym.errors import EnvDepsMissingError
    from lite.gym.registry import ensure_registered
    try:
        ensure_registered("browsergym")
    except EnvDepsMissingError as e:
        pytest.skip(f"browsergym deps not installed: {e.what}")

    for sub in ("miniwob", "webarena", "visualwebarena"):
        env_id = f"browsergym.{sub}"
        # Don't fail if a single sub-env's optional dep is missing — just
        # verify that, when registered, it's under the dotted form.
        if env_id in _splits:
            assert any(_splits[env_id].values()), (
                f"{env_id} registered but has 0 tasks in any split"
            )

    # At least one of the four MUST be registered (otherwise browsergym/main.py
    # didn't import OR all 4 registrations broke).
    registered_subs = [e for e in _splits if e.startswith("browsergym.")]
    assert registered_subs, (
        "No browsergym.* sub-envs registered. browsergym/main.py umbrella "
        "may not be importing dependencies, or the env_id rename to dotted "
        "form is broken."
    )


def test_no_legacy_hyphen_env_ids_in_registry():
    """No env_id with a hyphen should appear in the registry after the rename.

    ``-`` is forbidden in static names (env_ids etc.).
    Sweeps for both ``lite-*`` (renamed in PR1) and ``browsergym-*`` (renamed
    in PR2 follow-up).
    """
    # Force-load all envs so _splits is fully populated.
    _import_all()
    legacy = [
        k for k in _splits
        if k.startswith("lite-") or k.startswith("browsergym-")
    ]
    assert not legacy, (
        f"Legacy hyphen-form env_ids found in registry: {legacy}. "
        f"Should have been renamed to dotted form."
    )


def _fake_envs_dir(tmp_path, names):
    """A stand-in ``_ENVS_DIR`` holding one ``<name>/main.py`` per name."""
    for name in names:
        (tmp_path / name).mkdir()
        (tmp_path / name / "main.py").write_text("")
    return tmp_path


def test_import_all_skips_only_the_typed_unavailable_conditions(tmp_path, monkeypatch):
    """``_import_all`` may skip an env only for TYPED reasons.

    ``EnvDepsMissingError`` (local: this host lacks the env's deps/materials),
    ``EnvUnavailable`` (remote: that server does not serve this env_id) and
    ``EnvServerUnavailable`` (remote: no usable server at all) all mean
    "legitimately not available here", and all leave the *rest* of the sweep
    running — the property ``registered_env_ids`` depends on to report
    ``available: false`` with an install hint.
    """
    import importlib

    from lite.gym.errors import (
        EnvDepsMissingError,
        EnvServerUnavailable,
        EnvUnavailable,
    )

    reg = importlib.import_module("lite.gym.registry")
    monkeypatch.setattr(
        reg,
        "_ENVS_DIR",
        _fake_envs_dir(tmp_path, ["a_deps", "b_unserved", "c_no_server", "d_fine"]),
    )

    seen: list[str] = []

    def _fake_import_env(name, server_url=None):
        seen.append(name)
        if name == "a_deps":
            raise EnvDepsMissingError(what="no deps here", install="./install.sh", see="docs")
        if name == "b_unserved":
            raise EnvUnavailable("server doesn't serve it", status_code=404)
        if name == "c_no_server":
            raise EnvServerUnavailable("env-server unreachable at http://dead:1")

    monkeypatch.setattr(reg, "_import_env", _fake_import_env)

    reg._import_all()  # must NOT raise
    assert seen == ["a_deps", "b_unserved", "c_no_server", "d_fine"]


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("any other bug"),
        ImportError("circular import inside the env package"),
        ModuleNotFoundError("no module named 'some_internal_helper'"),
        SyntaxError("main.py does not parse"),
        KeyError("bad task data"),
    ],
    ids=["runtime", "import", "module-not-found", "syntax", "key"],
)
def test_import_all_never_reports_a_broken_env_as_absent(tmp_path, monkeypatch, exc):
    """A BROKEN env must be loud — never demoted to "this env isn't here".

    ``_import_all`` used to wrap the whole per-env import in
    ``except Exception as e: logger.debug("Skipping env '%s' (%s)", ...)``, so a
    syntax error, a circular import or a contradictory committed lock all
    vanished from the catalog with no message at default log level: the handler
    chose the diagnosis, and the diagnosis was always "absent". Only the typed
    conditions above are allowed to mean that.
    """
    import importlib

    reg = importlib.import_module("lite.gym.registry")
    monkeypatch.setattr(reg, "_ENVS_DIR", _fake_envs_dir(tmp_path, ["broken"]))

    def _fake_import_env(name, server_url=None):
        raise exc

    monkeypatch.setattr(reg, "_import_env", _fake_import_env)

    with pytest.raises(type(exc)):
        reg._import_all()


def test_import_all_propagates_a_committed_data_contradiction(tmp_path, monkeypatch):
    """``LiteContractError`` — the type a drifted COMMITTED lock raises — is NOT a
    host condition, so it must escape the catalog sweep.

    Retyping the lock failure away from ``EnvDepsMissingError`` (so a fresh clone
    fails loudly instead of skipping) only half-worked while ``_import_all`` still
    turned every escaping exception into ``logger.debug`` + a missing env.
    """
    import importlib

    from lite.core.errors import LiteContractError

    reg = importlib.import_module("lite.gym.registry")
    monkeypatch.setattr(reg, "_ENVS_DIR", _fake_envs_dir(tmp_path, ["drifted"]))

    def _fake_import_env(name, server_url=None):
        raise LiteContractError("catalog lock disagrees with what it pins")

    monkeypatch.setattr(reg, "_import_env", _fake_import_env)

    with pytest.raises(LiteContractError):
        reg._import_all()


def test_route_target_server_never_routes_to_itself():
    """The direct/remote routing policy, as a pure truth table.

    Regression for the env-server self-call deadlock: the server commonly
    inherits the *client* var ``CUA_LITE_ENV_SERVER_URL`` (e.g. from ``.zshrc``).
    The policy must report "no remote" for the server regardless of that var, so
    routing keys on the explicit server role — not the var's presence. Holds
    identically whether server and client share a host or not (role, not machine).
    """
    from lite.gym.utils.server.routing import _route_target

    # client + ambient URL set → route remote
    assert _route_target(is_server=False, ambient_url="http://s:30100") == "http://s:30100"
    # server → local even with the var set (the self-route the fix prevents)
    assert _route_target(is_server=True, ambient_url="http://s:30100") is None
    # client + nothing set → local (plain direct mode)
    assert _route_target(is_server=False, ambient_url=None) is None


def test_single_flight_coalesces_concurrent_imports(monkeypatch):
    """N concurrent first-imports of the same env coalesce to ONE remote
    ``register_from_server`` fetch — the cold-start thundering-herd fix. A
    small sleep in the fake fetch widens the lock-hold window so the peer
    threads must pile on the single-flight lock (not run serially)."""
    import threading
    import time

    import lite.gym.remote.client as rc

    env_id = "webgym"
    _reset_registry_state(env_id)
    try:
        monkeypatch.setenv("CUA_LITE_ENV_SERVER_URL", "http://test-server:30100")

        calls: list[str] = []
        calls_lock = threading.Lock()

        def _fake_fetch(name, url):
            with calls_lock:
                calls.append(name)
            time.sleep(0.05)  # hold the lock long enough for peers to coalesce

        monkeypatch.setattr(rc, "register_from_server", _fake_fetch)

        n = 16
        start = threading.Barrier(n)
        errors: list[Exception] = []

        def _worker():
            start.wait()  # release all threads together → maximize the race
            try:
                _import_env(env_id)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=_worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert calls == [env_id]                  # exactly ONE fetch for N callers
        assert _imported.get(env_id) == "remote"  # warm cache after the single fetch
    finally:
        _reset_registry_state(env_id)


def test_failed_fetch_does_not_poison_cache(monkeypatch):
    """A failed ``register_from_server`` must leave ``_imported`` unset so a
    retry re-fetches — the fix sets the cache ONLY on success."""
    import lite.gym.remote.client as rc

    env_id = "webgym"
    _reset_registry_state(env_id)
    try:
        monkeypatch.setenv("CUA_LITE_ENV_SERVER_URL", "http://test-server:30100")

        state = {"calls": 0}

        def _flaky_fetch(name, url):
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("transient server error")

        monkeypatch.setattr(rc, "register_from_server", _flaky_fetch)

        with pytest.raises(RuntimeError):
            _import_env(env_id)
        assert _imported.get(env_id) is None      # failure did not poison the cache

        _import_env(env_id)                        # retry re-attempts and succeeds
        assert state["calls"] == 2
        assert _imported.get(env_id) == "remote"
    finally:
        _reset_registry_state(env_id)


def test_failed_remote_refresh_preserves_warm_local_catalog(monkeypatch):
    """A transient remote refresh must not erase a working direct catalog."""
    import lite.gym.remote.client as rc

    env_id = "lite.demo"
    _reset_registry_state(env_id)
    try:
        monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
        _import_env(env_id)
        local_keys = {key for key in _specs if key.startswith(f"{env_id}@")}
        assert local_keys
        assert _imported.get(env_id) == "local"

        def _failed_fetch(name, url):
            raise RuntimeError("transient server error")

        monkeypatch.setattr(rc, "register_from_server", _failed_fetch)
        with pytest.raises(RuntimeError):
            _import_env(env_id, server_url="http://server-a")

        assert _imported.get(env_id) == "local"
        assert local_keys <= {key for key in _specs if key.startswith(f"{env_id}@")}

        _import_env(env_id)
        assert _imported.get(env_id) == "local"
        assert local_keys <= {key for key in _specs if key.startswith(f"{env_id}@")}
    finally:
        _reset_registry_state(env_id)


def test_failed_parent_remote_refresh_preserves_dotted_child_specs(monkeypatch):
    """A failed parent namespace fetch must not drop registered dotted children."""
    import lite.gym.remote.client as rc

    parent = "faketest_parent"
    child = f"{parent}.child"
    key = f"{child}@t1"
    _reset_registry_state(parent)
    _reset_registry_state(child)
    try:
        _splits[child] = {"eval": ["t1"]}
        _specs[key] = object()
        _imported[child] = "local"

        def _failed_fetch(name, url):
            raise RuntimeError("parent namespace is not makeable")

        monkeypatch.setattr(rc, "register_from_server", _failed_fetch)
        with pytest.raises(RuntimeError):
            _import_env(parent, server_url="http://server-a")

        assert _splits[child] == {"eval": ["t1"]}
        assert _specs[key] is not None
        assert _imported.get(child) == "local"
        assert _imported.get(parent) is None
    finally:
        _reset_registry_state(parent)
        _reset_registry_state(child)


def test_route_mode_flip_clears_env_make_kwargs(monkeypatch):
    """Remote stubs and local imports must not share env-wide make kwargs.

    A remote /tasks payload can carry env_make_kwargs that differ from the
    local config. When CUA_LITE_ENV_SERVER_URL is later unset, the direct import
    must start from the local source of truth instead of inheriting the remote
    stub defaults.
    """
    env_id = "lite.demo"  # local config intentionally has no make_kwargs block.
    remote_task = "remote_only"
    _reset_registry_state(env_id)
    try:
        monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
        _imported[env_id] = "remote"
        _splits[env_id] = {"eval": [remote_task]}
        _specs[f"{env_id}@{remote_task}"] = object()
        _env_make_kwargs[env_id] = {"cursor": False, "step_timeout": 9.0}
        _env_supported_kwargs[env_id] = {"seed"}

        _import_env(env_id)

        assert _imported.get(env_id) == "local"
        assert _env_make_kwargs.get(env_id, {}) == {}
        assert _env_supported_kwargs.get(env_id, set()) == set()
        assert f"{env_id}@{remote_task}" not in _specs
        assert any(_specs_key.startswith(f"{env_id}@") for _specs_key in _specs)
    finally:
        _reset_registry_state(env_id)


def test_explicit_env_server_url_routes_before_local_import(monkeypatch):
    """A per-call env_server_url must fetch remote specs without local deps."""
    import lite.gym.remote as remote
    from lite.core import LiteCUAMetadata
    from lite.gym.registry import register
    from lite.gym.remote import client as remote_client

    env_id = "faketest_explicit_remote"
    key = f"{env_id}@t1"
    captured: dict[str, object] = {}
    _reset_registry_state(env_id)

    class FakeClient:
        metadata_or_none = None

        def __init__(self, server_url, env_key, **kwargs):
            captured["server_url"] = server_url
            captured["env_key"] = env_key
            captured["kwargs"] = kwargs

    def fake_register_from_server(name: str, server_url: str) -> None:
        captured["fetched"] = (name, server_url)
        register(
            key,
            entry_point=_carry_stub_entry_point,
            split="eval",
            metadata=LiteCUAMetadata(dims=("desktop", "use")),
            cursor=False,
        )

    try:
        monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
        monkeypatch.setattr(remote_client, "register_from_server", fake_register_from_server)
        monkeypatch.setattr(remote, "LiteEnvClient", FakeClient)

        env = gym.make(key, env_server_url="http://server-a")

        assert isinstance(env, FakeClient)
        assert captured["fetched"] == (env_id, "http://server-a")
        assert captured["server_url"] == "http://server-a"
        assert captured["env_key"] == key
        assert captured["kwargs"]["cursor"] is False
    finally:
        _reset_registry_state(env_id)


def test_remote_import_cache_keys_on_server_url(monkeypatch):
    """Switching env-server URLs must not reuse stale remote stub specs."""
    from lite.core import LiteCUAMetadata
    from lite.gym.registry import register
    from lite.gym.remote import client as remote_client

    env_id = "faketest_remote_url_switch"
    calls: list[str] = []
    _reset_registry_state(env_id)

    def fake_register_from_server(name: str, server_url: str) -> None:
        calls.append(server_url)
        register(
            f"{name}@t1",
            entry_point=_carry_stub_entry_point,
            split="eval",
            metadata=LiteCUAMetadata(dims=("desktop", "use")),
            cursor=server_url.endswith("b"),
        )

    try:
        monkeypatch.setattr(remote_client, "register_from_server", fake_register_from_server)

        _import_env(env_id, server_url="http://server-a")
        assert _specs[f"{env_id}@t1"].kwargs["cursor"] is False
        _import_env(env_id, server_url="http://server-b")

        assert calls == ["http://server-a", "http://server-b"]
        assert _imported.get(env_id) == "remote"
        assert _imported_remote_urls.get(env_id) == "http://server-b"
        assert _specs[f"{env_id}@t1"].kwargs["cursor"] is True
    finally:
        _reset_registry_state(env_id)


def test_local_remote_local_reimports_cached_local_module(monkeypatch):
    """Switching back to direct mode must replay local registration side effects."""
    from lite.core import LiteCUAMetadata
    from lite.gym.registry import register
    from lite.gym.remote import client as remote_client

    env_id = "lite.demo"
    remote_key = f"{env_id}@remote_only"
    _reset_registry_state(env_id)

    def fake_register_from_server(name: str, server_url: str) -> None:
        register(
            f"{name}@remote_only",
            entry_point=_carry_stub_entry_point,
            split="eval",
            metadata=LiteCUAMetadata(dims=("desktop", "use")),
            cursor=False,
        )

    try:
        monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
        monkeypatch.setattr(remote_client, "register_from_server", fake_register_from_server)

        _import_env(env_id)
        local_keys = {key for key in _specs if key.startswith(f"{env_id}@")}
        assert local_keys
        assert _imported.get(env_id) == "local"

        _import_env(env_id, server_url="http://server-a")
        assert set(_specs) & local_keys == set()
        assert remote_key in _specs
        assert _imported.get(env_id) == "remote"

        _import_env(env_id)
        restored_keys = {key for key in _specs if key.startswith(f"{env_id}@")}
        assert _imported.get(env_id) == "local"
        assert _imported_remote_urls.get(env_id) is None
        assert remote_key not in _specs
        assert local_keys <= restored_keys
    finally:
        _reset_registry_state(env_id)


def test_remote_import_clears_transitive_child_specs_without_import_marker(monkeypatch):
    """Umbrella-imported child specs can exist before the child is marked imported."""
    from lite.core import LiteCUAMetadata
    from lite.gym.registry import register
    from lite.gym.remote import client as remote_client

    env_id = "faketest.parent.child"
    local_key = f"{env_id}@local_only"
    remote_key = f"{env_id}@remote_only"
    _reset_registry_state(env_id)

    def fake_register_from_server(name: str, server_url: str) -> None:
        register(
            f"{name}@remote_only",
            entry_point=_carry_stub_entry_point,
            split="eval",
            metadata=LiteCUAMetadata(dims=("desktop", "use")),
            cursor=True,
        )

    try:
        register(
            local_key,
            entry_point=_carry_stub_entry_point,
            split="eval",
            metadata=LiteCUAMetadata(dims=("desktop", "use")),
            cursor=False,
        )
        assert _imported.get(env_id) is None
        monkeypatch.setattr(remote_client, "register_from_server", fake_register_from_server)

        _import_env(env_id, server_url="http://server-a")

        assert local_key not in _specs
        assert remote_key in _specs
        assert _splits[env_id] == {"eval": ["remote_only"]}
        assert _specs[remote_key].kwargs["cursor"] is True
    finally:
        _reset_registry_state(env_id)


def test_parent_remote_import_clears_dotted_child_specs(monkeypatch):
    """Refreshing an umbrella parent must clear child specs it registered earlier."""
    from lite.core import LiteCUAMetadata
    from lite.gym.registry import register
    from lite.gym.remote import client as remote_client

    parent_id = "faketest.parent"
    child_id = f"{parent_id}.child"
    local_key = f"{child_id}@local_only"
    remote_key = f"{child_id}@remote_only"
    _reset_registry_state(child_id)
    _reset_registry_state(parent_id)

    def fake_register_from_server(name: str, server_url: str) -> None:
        assert name == parent_id
        register(
            f"{name}.child@remote_only",
            entry_point=_carry_stub_entry_point,
            split="eval",
            metadata=LiteCUAMetadata(dims=("desktop", "use")),
            cursor=True,
        )

    try:
        register(
            local_key,
            entry_point=_carry_stub_entry_point,
            split="eval",
            metadata=LiteCUAMetadata(dims=("desktop", "use")),
            cursor=False,
        )
        assert _imported.get(parent_id) is None
        assert _has_env_registration(parent_id)
        monkeypatch.setattr(remote_client, "register_from_server", fake_register_from_server)

        _import_env(parent_id, server_url="http://server-a")

        assert local_key not in _specs
        assert remote_key in _specs
        assert _splits[child_id] == {"eval": ["remote_only"]}
        assert _specs[remote_key].kwargs["cursor"] is True
    finally:
        _reset_registry_state(child_id)
        _reset_registry_state(parent_id)


def test_mode_flip_clears_lazy_registry_and_env_module_latches(monkeypatch):
    """Clearing specs must also clear lazy task/service latches."""
    from lite.core import LiteCUAMetadata
    from lite.gym.registry import register
    from lite.gym.remote import client as remote_client

    env_id = "faketest_lazy"
    local_key = f"{env_id}@local_only"
    remote_key = f"{env_id}@remote_only"
    module_name = f"lite.gym.envs.{env_id}.main"
    _reset_registry_state(env_id)

    fake_module = types.ModuleType(module_name)
    fake_module._tasks_registered = True

    def fake_register_from_server(name: str, server_url: str) -> None:
        register(
            f"{name}@remote_only",
            entry_point=_carry_stub_entry_point,
            split="eval",
            metadata=LiteCUAMetadata(dims=("desktop", "use")),
        )

    try:
        register(
            local_key,
            entry_point=_carry_stub_entry_point,
            split="eval",
            metadata=LiteCUAMetadata(dims=("desktop", "use")),
        )
        _tasks_registered.add(env_id)
        _services_started.add(env_id)
        sys.modules[module_name] = fake_module
        monkeypatch.setattr(remote_client, "register_from_server", fake_register_from_server)

        _import_env(env_id, server_url="http://server-a")

        assert local_key not in _specs
        assert remote_key in _specs
        assert env_id not in _tasks_registered
        assert env_id not in _services_started
        assert fake_module._tasks_registered is False
    finally:
        sys.modules.pop(module_name, None)
        _reset_registry_state(env_id)


def test_explicit_remote_make_fails_fast_when_url_switches_before_snapshot(monkeypatch):
    """Mixed remote URLs for one env_id must not silently cross-contaminate specs."""
    from lite.core import LiteCUAMetadata
    from lite.gym.registry import register

    env_id = "faketest_mixed_remote_url"
    key = f"{env_id}@t1"
    _reset_registry_state(env_id)

    def fake_import(name: str, server_url: str | None = None) -> None:
        register(
            key,
            entry_point=_carry_stub_entry_point,
            split="eval",
            metadata=LiteCUAMetadata(dims=("desktop", "use")),
            cursor=True,
        )
        _imported[name] = "remote"
        _imported_remote_urls[name] = "http://server-b"

    try:
        monkeypatch.setattr(sys.modules["lite.gym.registry"], "_import_env", fake_import)

        with pytest.raises(RuntimeError, match="Remote env catalog.*switched"):
            gym.make(key, env_server_url="http://server-a")
    finally:
        _reset_registry_state(env_id)


def _carry_stub_entry_point(**_kwargs):  # mirrors register_from_server's stub; never constructed
    raise AssertionError("stub entry_point must not be constructed in this test")


def test_carried_spec_kwargs_survive_server_to_client_roundtrip():
    """Direct and server modes must resolve the SAME per-task make config.

    Regression for the divergence where the env-server task-list endpoint dropped
    ``EnvSpec.kwargs``: remote stub specs had empty kwargs, so ``make()`` reverted
    registered timeouts to its defaults (e.g. osworld step_timeout 180 -> 120,
    killing its heavy reward-eval step). It must also carry env-owned make config
    such as ``cursor``. The fix carries ``CARRIED_SPEC_KWARGS`` from server to
    client (``task_kwargs`` -> payload -> ``register_from_server``).
    """
    from lite.gym.registry import CARRIED_SPEC_KWARGS, register, registry

    key = "faketest_carry@t1"
    try:
        # Register like osworld/browser envs: carried wrapper config, carried
        # env-owned make config, and a non-carried entry_point kwarg.
        register(key, entry_point=_carry_stub_entry_point, split="train",
                 step_timeout=180.0, cursor=False, some_entrypoint_kwarg="X")

        # Server side: task_kwargs exposes ONLY the carried subset (not entry_point kwargs).
        carried = registry.task_kwargs("faketest_carry", "t1")
        assert carried == {"step_timeout": 180.0, "cursor": False}, carried
        assert "some_entrypoint_kwarg" not in carried
        assert set(carried) <= set(CARRIED_SPEC_KWARGS)

        # Client side (register_from_server-style): rebuild the stub spec WITH the carried
        # kwargs -> spec.kwargs carries the registered value, so make() resolves 180 (not
        # the 120 default). Without the fix the stub kwargs would be empty.
        _specs.pop(key, None)
        register(key, entry_point=_carry_stub_entry_point, split="train", **carried)
        assert _specs[key].kwargs.get("step_timeout") == 180.0
        assert _specs[key].kwargs.get("cursor") is False
    finally:
        _reset_registry_state("faketest_carry")


def test_carried_spec_kwargs_split_wrapper_and_env_owned_make_sources():
    """Wrapper pops and carried make kwargs stay deliberately separated."""
    from lite.gym.registry import (
        _ENV_MAKE_KWARGS,
        _WRAPPER_KWARG_DEFAULTS,
        CARRIED_SPEC_KWARGS,
    )

    assert CARRIED_SPEC_KWARGS == (
        tuple(_WRAPPER_KWARG_DEFAULTS) + tuple(sorted(_ENV_MAKE_KWARGS))
    )
    # Wrapper config is still single-sourced from the dict factory.make pops.
    assert tuple(_WRAPPER_KWARG_DEFAULTS) == (
        "step_timeout",
        "reset_timeout",
        "loop_detect",
        "loop_detect_terminal_status",
    )
    assert set(_WRAPPER_KWARG_DEFAULTS) < set(CARRIED_SPEC_KWARGS)
    # Env-owned make kwargs are carried across /tasks but are not wrapper-popped.
    assert _ENV_MAKE_KWARGS == frozenset({"cursor"})
    assert "cursor" in CARRIED_SPEC_KWARGS
    assert "cursor" not in _WRAPPER_KWARG_DEFAULTS
    # The known wrapper set + sane defaults (a renamed/removed kwarg trips here).
    assert _WRAPPER_KWARG_DEFAULTS == {
        "step_timeout": 120.0, "reset_timeout": 600.0,
        "loop_detect": 0,
        "loop_detect_terminal_status": "success",
    }


def test_status_sensitive_envs_pin_loop_detect_terminal_status_to_success():
    """Env configs own the score-sensitive loop terminal status policy."""
    config_paths = {
        "osworld": "lite/gym/envs/osworld/configs/default.yaml",
        "browsergym": "lite/gym/envs/browsergym/configs/default.yaml",
        "browsergym/isolation": "lite/gym/envs/browsergym/configs/isolation.yaml",
        "waa": "lite/gym/envs/waa/configs/default.yaml",
    }
    for env_id, relative_path in config_paths.items():
        data = yaml.safe_load(
            (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        ) or {}
        assert data["make_kwargs"]["loop_detect_terminal_status"] == "success", env_id


@pytest.mark.parametrize(
    ("step_timeout", "reset_timeout", "expected"),
    [(1800.0, 600.0, 1860.0), (None, 600.0, 660.0)],
)
def test_remote_rpc_timeout_covers_server_wrapper(
    monkeypatch, step_timeout, reset_timeout, expected
):
    """The HTTP client must outlive whichever server-side timeout is longer."""
    import lite.gym.remote as remote
    from lite.gym.registry import register

    registry_module = sys.modules["lite.gym.registry"]
    env_id = "faketest_rpc_timeout"
    key = f"{env_id}@t1"
    captured = {}

    class FakeClient:
        metadata_or_none = None  # factory reads the public nullable accessor

        def __init__(self, *args, timeout, **kwargs):
            captured["timeout"] = timeout
            self._metadata = None

    try:
        register(key, entry_point=_carry_stub_entry_point)
        _imported[env_id] = "remote"
        _imported_remote_urls[env_id] = "http://test-server"
        monkeypatch.setattr(registry_module, "_import_env", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(remote, "LiteEnvClient", FakeClient)

        gym.make(
            key,
            env_server_url="http://test-server",
            step_timeout=step_timeout,
            reset_timeout=reset_timeout,
        )
        assert captured["timeout"] == expected
    finally:
        _reset_registry_state(env_id)


def test_remote_mode_does_not_apply_client_timeout_wrapper(monkeypatch):
    """Remote mode lets the server own reset/step timeout enforcement."""
    import lite.gym.remote as remote
    from lite.gym.registry import register
    from lite.gym.wrappers import StepTimeoutWrapper

    registry_module = sys.modules["lite.gym.registry"]
    env_id = "faketest_remote_timeout_owner"
    key = f"{env_id}@t1"

    class FakeClient:
        metadata_or_none = None

        def __init__(self, *_args, **_kwargs):
            pass

    try:
        register(key, entry_point=_carry_stub_entry_point)
        _imported[env_id] = "remote"
        _imported_remote_urls[env_id] = "http://test-server"
        monkeypatch.setattr(registry_module, "_import_env", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(remote, "LiteEnvClient", FakeClient)

        env = gym.make(
            key,
            env_server_url="http://test-server",
            step_timeout=5.0,
            reset_timeout=7.0,
        )

        assert isinstance(env, FakeClient)
        assert not isinstance(env, StepTimeoutWrapper)
    finally:
        _reset_registry_state(env_id)


def test_register_from_server_version_skew_fails_loud(monkeypatch):
    """section B6.2: a /tasks payload missing the required 'splits' key raises the
    single clear version-mismatch error (was a bare KeyError); optional keys
    keep their empty fallbacks."""
    import pytest as _pytest

    from lite.gym.remote import client as client_mod
    from lite.gym.remote.client import register_from_server

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"metadata": {}, "kwargs": {}}   # no 'splits'

    monkeypatch.setattr(client_mod.httpx, "get", lambda *a, **k: _Resp())
    with _pytest.raises(RuntimeError, match="version mismatch"):
        register_from_server("someenv", "http://fake:1")


def test_register_from_server_preserves_env_supported_kwargs(monkeypatch):
    """Remote registration must preserve env-owned soft-kwarg capabilities."""
    from lite.gym.remote import client as client_mod
    from lite.gym.remote.client import register_from_server

    env_id = "faketest_supported_kwargs"
    _reset_registry_state(env_id)

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "splits": {"train": ["t1"]},
                "metadata": {},
                "kwargs": {},
                "env_make_kwargs": {},
                "env_supported_kwargs": ["seed", "cursor"],
            }

    try:
        monkeypatch.setattr(client_mod.httpx, "get", lambda *a, **k: _Resp())
        register_from_server(env_id, "http://fake:1")

        assert gym.registry.env_supported_kwargs(env_id) == ["cursor", "seed"]
        assert gym.registry.env_supports_kwarg(env_id, "seed") is True
    finally:
        _reset_registry_state(env_id)


def test_register_from_server_preserves_env_make_kwargs(monkeypatch):
    """Remote registration must preserve env-wide ``make_kwargs`` defaults."""
    from lite.gym.remote import client as client_mod
    from lite.gym.remote.client import register_from_server

    env_id = "faketest_make_kwargs"
    _reset_registry_state(env_id)

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "splits": {"train": ["t1"]},
                "metadata": {},
                "kwargs": {},
                "env_make_kwargs": {"cursor": False, "step_timeout": 7.0},
                "env_supported_kwargs": [],
            }

    try:
        monkeypatch.setattr(client_mod.httpx, "get", lambda *a, **k: _Resp())
        register_from_server(env_id, "http://fake:1")

        assert gym.registry.env_make_kwargs(env_id) == {
            "cursor": False,
            "step_timeout": 7.0,
        }
    finally:
        _reset_registry_state(env_id)


def test_register_from_server_parses_generic_task_metadata(monkeypatch):
    """Remote /tasks metadata is parsed by the client-side union reader."""
    from lite.core import LiteGenericMetadata
    from lite.gym.remote import client as client_mod
    from lite.gym.remote.client import register_from_server

    env_id = "faketest_generic_from_server"
    _reset_registry_state(env_id)
    remote_metadata = LiteGenericMetadata(others={"source": "server"}).to_dict()

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "splits": {"train": ["t1"]},
                "metadata": {"t1": remote_metadata},
                "kwargs": {},
                "env_make_kwargs": {},
                "env_supported_kwargs": [],
            }

    try:
        monkeypatch.setattr(client_mod.httpx, "get", lambda *a, **k: _Resp())
        register_from_server(env_id, "http://fake:1")

        meta = gym.registry.task_metadata(env_id, "t1")
        assert isinstance(meta, LiteGenericMetadata)
        assert meta.dims == ()
        assert meta.others == {
            "source": "server",
            "env_id": env_id,
            "task_id": "t1",
        }
    finally:
        _reset_registry_state(env_id)


def test_make_does_not_install_tool_availability_gate(monkeypatch):
    """Runtime unsupported/invalid feedback is owned by concrete env.step."""
    from lite.core import LiteCUAMetadata
    from lite.gym.base import LiteBaseEnv
    from lite.gym.registry import register

    class _StubEnv(LiteBaseEnv):
        def _runtime_metadata(self) -> LiteCUAMetadata:
            return LiteCUAMetadata(
                dims=("desktop", "use"),
                extra_tool_schemas=[],
            )

        async def reset(self):
            raise AssertionError("not reached")

        async def step(self, actions):
            raise AssertionError("not reached")

        async def close(self):
            pass

    registry_module = sys.modules["lite.gym.registry"]
    env_id = "faketest_gate_outermost"
    key = f"{env_id}@t1"
    try:
        register(key, entry_point=lambda **_kw: _StubEnv())
        monkeypatch.setattr(registry_module, "_import_env", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(registry_module, "ensure_services", lambda _: None)

        env = gym.make(key, env_server_url=None, step_timeout=None)
        assert isinstance(env, _StubEnv)
    finally:
        _specs.pop(key, None)
        _splits.pop(env_id, None)


@pytest.mark.parametrize(
    ("loop_detect", "terminal_status", "expect_detector"),
    [(0, "failure", False), (5, "success", True), (5, "failure", True)],
    ids=["default_off", "success_policy", "failure_policy"],
)
def test_make_wires_loop_detector_without_tool_gate(
    monkeypatch, loop_detect, terminal_status, expect_detector,
):
    """Loop detection is the outer wrapper when enabled; no tool gate is installed."""
    from lite.core import LiteCUAMetadata
    from lite.gym.base import LiteBaseEnv
    from lite.gym.registry import register
    from lite.gym.wrappers import LoopDetectWrapper

    class _StubEnv(LiteBaseEnv):
        def _runtime_metadata(self) -> LiteCUAMetadata:
            return LiteCUAMetadata(
                dims=("desktop", "use"),
                extra_tool_schemas=[],
            )

        async def reset(self):
            raise AssertionError("not reached")

        async def step(self, actions):
            raise AssertionError("not reached")

        async def close(self):
            pass

    registry_module = sys.modules["lite.gym.registry"]
    env_id = "faketest_gate_loop_wiring"
    key = f"{env_id}@t1"
    try:
        def entry_point(**kw):
            assert "loop_detect" not in kw
            assert "loop_detect_terminal_status" not in kw
            return _StubEnv()

        register(key, entry_point=entry_point)
        monkeypatch.setattr(registry_module, "_import_env", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(registry_module, "ensure_services", lambda _: None)

        kwargs = {"loop_detect_terminal_status": terminal_status}
        if loop_detect:
            kwargs["loop_detect"] = loop_detect
        env = gym.make(
            key, env_server_url=None, step_timeout=None,
            **kwargs,
        )
        if expect_detector:
            assert isinstance(env, LoopDetectWrapper)
            assert env._terminal_status == terminal_status
            assert isinstance(env.env, _StubEnv)
        else:
            assert isinstance(env, _StubEnv)
    finally:
        _specs.pop(key, None)
        _splits.pop(env_id, None)
