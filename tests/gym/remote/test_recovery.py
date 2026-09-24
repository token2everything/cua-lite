"""A scoped server imports and cleans only its explicitly served env IDs."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from lite.gym.errors import EnvDepsMissingError
from lite.gym.remote import recovery
from lite.gym.remote.scope import ServerScope
from lite.gym.services import BackendFamily, EnvServices
from lite.gym.types import StateView

SCOPE = ServerScope(server_port=30100, token_hash="test-owned")


@pytest.fixture(params=["recover", "reconcile", "shutdown"])
def sweep(request):
    if request.param == "reconcile":
        return lambda allowed: recovery.reconcile_all(
            StateView(by_env_id={}, snapshot_at=0.0), SCOPE, allowed_env_ids=allowed
        )
    return lambda allowed: getattr(recovery, request.param + "_all")(SCOPE, allowed_env_ids=allowed)


def test_scoped_sweeps_exclude_broken_modules_and_sibling_services(monkeypatch, sweep):
    calls = []

    class SharedServices(EnvServices):
        def live_ids(self, env_id, scope):
            calls.append(("live_ids", env_id))
            return set()

        def reap(self, env_id, scope, in_use, *, boot=False):
            calls.append(("reap", env_id))
            return 0

        def shutdown(self, env_id, scope):
            calls.append(("shutdown", env_id))

    # The lazy selected leaf has no task specs yet. Its service is registered
    # on the umbrella; a sibling is registered but is not served by this app.
    monkeypatch.setattr(recovery.registry, "_specs", {"family.sibling@task": object()})
    monkeypatch.setattr(
        recovery.registry, "_services", {"family": SharedServices(), "broken": EnvServices()}
    )
    monkeypatch.setattr(recovery.registry, "_families", {"family": BackendFamily.DEDICATED})
    monkeypatch.setattr(
        recovery.registry, "_import_all", Mock(side_effect=RuntimeError("unrelated broken module"))
    )

    def import_selected(env_id):
        if env_id != "family.selected":
            raise RuntimeError("unrelated broken module")

    monkeypatch.setattr(recovery.registry, "_import_env", import_selected)
    sweep({"family.selected"})
    assert calls
    assert {env_id for _, env_id in calls} == {"family.selected"}


def test_empty_allowlist_never_imports_or_cleans_registered_services(monkeypatch, sweep):
    monkeypatch.setattr(recovery.registry, "_services", {"broken": EnvServices()})
    monkeypatch.setattr(recovery.registry, "_families", {})
    broad = Mock(side_effect=AssertionError("broad import must not run"))
    selected = Mock(side_effect=AssertionError("selected import must not run"))
    monkeypatch.setattr(recovery.registry, "_import_all", broad)
    monkeypatch.setattr(recovery.registry, "_import_env", selected)
    assert sweep(set()) in (0, [])
    broad.assert_not_called()
    selected.assert_not_called()


def test_unscoped_sweeps_still_report_broken_catalog_imports(monkeypatch, sweep):
    monkeypatch.setattr(
        recovery.registry, "_import_all", Mock(side_effect=RuntimeError("broken catalog"))
    )
    with pytest.raises(RuntimeError, match="broken catalog"):
        sweep(None)


def test_scoped_sweeps_do_not_hide_selected_module_errors(monkeypatch, sweep):
    monkeypatch.setattr(
        recovery.registry, "_import_env", Mock(side_effect=RuntimeError("broken selected env"))
    )
    with pytest.raises(RuntimeError, match="broken selected env"):
        sweep({"selected"})


def test_unavailable_selected_environment_has_no_lifecycle(monkeypatch, sweep):
    monkeypatch.setattr(
        recovery.registry,
        "_import_env",
        Mock(side_effect=EnvDepsMissingError(what="not installed", install="test", see="test")),
    )
    assert sweep({"unavailable"}) in (0, [])
