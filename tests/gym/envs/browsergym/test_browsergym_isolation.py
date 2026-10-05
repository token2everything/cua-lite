"""BrowserGym WebArena/VisualWebArena isolation and service lifecycle tests."""

from __future__ import annotations

import io
import os
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("browsergym.core", reason="browsergym not installed")

from lite.gym.envs.browsergym.main import (
    _DEFAULT_ENV_VARS,
    _check_env_vars,
    _wa_vwa_task_facts,
)

# ---------------------------------------------------------------------------
# WA / VWA static task facts (sites + llm_as_a_judge)
# ---------------------------------------------------------------------------


class TestWaVwaTaskFacts:
    """``_wa_vwa_task_facts`` returns STATIC task facts ``(sites, llm_judge,
    mutating)`` from the raw upstream config — no reachability probing (retired
    with the ``exclude_reason`` baking; callers compose their own runtime
    filters). Uses cache patching to avoid disk reads."""

    def setup_method(self):
        from lite.gym.envs.browsergym import isolation

        isolation._RAW_CONFIG_CACHE.clear()
        isolation._DEPENDS_ON_CACHE.clear()

    def teardown_method(self):
        from lite.gym.envs.browsergym import isolation

        isolation._RAW_CONFIG_CACHE.clear()
        isolation._DEPENDS_ON_CACHE.clear()

    def _seed_cache(self, benchmark: str, tasks: dict[str, dict]) -> None:
        """Bypass the disk JSON load by pre-populating the shared cache
        (``isolation.task_raw_config`` reads this)."""
        from lite.gym.envs.browsergym import isolation

        isolation._RAW_CONFIG_CACHE[benchmark] = tasks

    def test_unknown_benchmark_returns_empty(self):
        assert _wa_vwa_task_facts("miniwob", "miniwob.click-dialog") == ([], False, False)

    def test_task_not_in_cache_returns_empty(self):
        self._seed_cache("webarena", {})
        assert _wa_vwa_task_facts("webarena", "webarena.999") == ([], False, False)

    def test_plain_shopping_task(self):
        # string_match (exact_match reference answer) → read-only Q&A → not mutating.
        self._seed_cache(
            "webarena",
            {
                "webarena.0": {
                    "sites": ["shopping"],
                    "eval": {"reference_answers": {"exact_match": "x"}},
                },
            },
        )
        assert _wa_vwa_task_facts("webarena", "webarena.0") == (["shopping"], False, False)

    def test_map_site_surfaced_not_excluded(self):
        # ``map`` is surfaced as a static site fact; NOT probed/excluded here —
        # the caller filters it out at runtime (e.g. when OSM isn't provisioned).
        self._seed_cache(
            "webarena",
            {
                "webarena.5": {"sites": ["map"], "eval": {}},
            },
        )
        assert _wa_vwa_task_facts("webarena", "webarena.5") == (["map"], False, False)

    def test_classifieds_site_surfaced(self):
        # classifieds is a real VWA service; surfaced as a site fact, never
        # reachability-probed at registration (that mutable check was retired).
        self._seed_cache(
            "visualwebarena",
            {
                "visualwebarena.5": {"sites": ["classifieds"], "eval": {}},
            },
        )
        assert _wa_vwa_task_facts("visualwebarena", "visualwebarena.5") == (
            ["classifieds"],
            False,
            False,
        )

    def test_llm_judge_fuzzy_match(self):
        self._seed_cache(
            "webarena",
            {
                "webarena.7": {
                    "sites": ["shopping"],
                    "eval": {"reference_answers": {"fuzzy_match": ["something"]}},
                },
            },
        )
        assert _wa_vwa_task_facts("webarena", "webarena.7") == (["shopping"], True, False)

    def test_sites_sorted_and_deduped_shape(self):
        self._seed_cache(
            "webarena",
            {
                "webarena.8": {"sites": ["reddit", "gitlab"], "eval": {}},
            },
        )
        sites, llm_judge, mutating = _wa_vwa_task_facts("webarena", "webarena.8")
        assert sites == ["gitlab", "reddit"]  # sorted
        assert llm_judge is False
        assert mutating is False

    def test_mutating_write_task(self):
        # A ``program_html`` eval that re-navigates to a SPECIFIC url (not "last")
        # verifies PERSISTED state → the task WROTE the shared backend → mutating.
        self._seed_cache(
            "webarena",
            {
                "webarena.10": {
                    "sites": ["gitlab"],
                    "intent": "Create a new repository named webagent",
                    "eval": {
                        "program_html": [{"url": "__GITLAB__/webagent", "required_contents": {}}]
                    },
                },
            },
        )
        assert _wa_vwa_task_facts("webarena", "webarena.10") == (["gitlab"], False, True)

    def test_read_only_string_match_not_mutating(self):
        # Pure string_match Q&A ("how many ...") → no persisted write → not mutating.
        self._seed_cache(
            "webarena",
            {
                "webarena.11": {
                    "sites": ["shopping_admin"],
                    "intent": "How many items were sold in the most recent order?",
                    "eval": {"reference_answers": {"exact_match": "7"}},
                },
            },
        )
        assert _wa_vwa_task_facts("webarena", "webarena.11") == (["shopping_admin"], False, False)

    def test_map_plus_llm_judge_both_surfaced(self):
        # Both facts coexist independently — no comma-joined exclude_reason string.
        self._seed_cache(
            "webarena",
            {
                "webarena.99": {
                    "sites": ["map"],
                    "eval": {"reference_answers": {"fuzzy_match": ["x"]}},
                },
            },
        )
        assert _wa_vwa_task_facts("webarena", "webarena.99") == (["map"], True, False)

    def test_verified_uses_official_mutation_type_without_llm_judge(self, monkeypatch):
        from lite.gym.envs.browsergym import webarena_verified

        task_id = "webarena_verified.303.44.2"
        self._seed_cache(
            "webarena_verified",
            {
                task_id: {
                    "sites": ["gitlab"],
                    "intent": "Create a repository",
                    "eval": [
                        {
                            "evaluator": "AgentResponseEvaluator",
                            "expected": {"task_type": "mutate"},
                        }
                    ],
                }
            },
        )
        monkeypatch.setattr(
            webarena_verified,
            "is_mutating",
            lambda candidate: candidate == task_id,
        )

        assert _wa_vwa_task_facts("webarena_verified", task_id) == (
            ["gitlab"],
            False,
            True,
        )

    def test_verified_metadata_surfaces_hard_and_dependency_facts(self, monkeypatch):
        from lite.gym.envs.browsergym import isolation, webarena_verified
        from lite.gym.envs.browsergym.main import BrowserGymEnv

        task_id = "webarena_verified.303.44.2"
        parent = "webarena_verified.285.43.2"
        self._seed_cache(
            "webarena_verified",
            {task_id: {"sites": ["gitlab"], "intent": "Create", "eval": []}},
        )
        isolation._DEPENDS_ON_CACHE["webarena_verified"] = {task_id: [parent]}
        monkeypatch.setattr(webarena_verified, "is_mutating", lambda candidate: True)
        monkeypatch.setattr(webarena_verified, "is_hard", lambda candidate: True)

        metadata = BrowserGymEnv._task_metadata(
            "webarena_verified",
            task_id,
            action_subsets=("coord", "chat"),
            viewport=(1280, 720),
        )

        assert metadata.others["sites"] == ["gitlab"]
        assert metadata.others["mutating"] is True
        assert metadata.others["hard"] is True
        assert metadata.others["depends_on"] == [parent]
        assert "llm_as_a_judge" not in metadata.others


# ---------------------------------------------------------------------------
# depends_on ingestion (topological run-order for the strict-isolation flow)
# ---------------------------------------------------------------------------


class TestTaskDependsOn:
    """``isolation.task_depends_on`` loads BrowserGym's curated ``depends_on``
    run-order. Cache-seeded for the unit cases; one real-CSV integration check."""

    def setup_method(self):
        from lite.gym.envs.browsergym import isolation

        isolation._DEPENDS_ON_CACHE.clear()

    def teardown_method(self):
        from lite.gym.envs.browsergym import isolation

        isolation._DEPENDS_ON_CACHE.clear()

    def test_seeded_parents(self):
        from lite.gym.envs.browsergym import isolation

        isolation._DEPENDS_ON_CACHE["webarena"] = {"webarena.1": ["webarena.0"], "webarena.0": []}
        assert isolation.task_depends_on("webarena", "webarena.1") == ["webarena.0"]
        assert isolation.task_depends_on("webarena", "webarena.0") == []  # root

    def test_miniwob_and_unknown_empty(self):
        from lite.gym.envs.browsergym import isolation

        isolation._DEPENDS_ON_CACHE["webarena"] = {}
        assert isolation.task_depends_on("miniwob", "miniwob.x") == []
        assert isolation.task_depends_on("webarena", "webarena.999") == []

    def test_real_csv_loads(self):
        # Integration: the bundled BrowserGym experiments metadata CSV resolves
        # without importing its pandas-dependent package and parses both task
        # families (task 1 → its curated task-0 parent).
        from lite.gym.envs.browsergym import isolation

        isolation._DEPENDS_ON_CACHE.clear()
        assert isolation.task_depends_on("webarena", "webarena.1") == ["webarena.0"]
        assert isolation.task_depends_on(
            "webarena_verified", "webarena_verified.279.1.2"
        ) == ["webarena_verified.279.0.2"]


# ---------------------------------------------------------------------------
# Mutating / write classification (conflict gate)
# ---------------------------------------------------------------------------


class TestMutatingClassification:
    """``isolation.is_mutating`` heuristic — synthetic unit cases for each
    signal, plus an oracle guard over the REAL upstream configs.

    Correctness > parallelism (section 11.7): a false negative (a write classed
    as a shareable reader) corrupts the shared stack, so the oracle's job
    is to catch *under*-marking regressions.
    """

    def setup_method(self):
        from lite.gym.envs.browsergym import isolation

        isolation._RAW_CONFIG_CACHE.clear()

    def teardown_method(self):
        from lite.gym.envs.browsergym import isolation

        isolation._RAW_CONFIG_CACHE.clear()

    # --- each signal fires independently ---
    def test_require_reset_signal(self):
        from lite.gym.envs.browsergym import isolation

        assert isolation.is_mutating({"require_reset": True})

    def test_program_html_renavigation_is_write(self):
        from lite.gym.envs.browsergym import isolation

        assert isolation.is_mutating({"eval": {"program_html": [{"url": "__GITLAB__/x/-/issues"}]}})

    def test_program_html_on_last_is_read(self):
        from lite.gym.envs.browsergym import isolation

        assert not isolation.is_mutating(
            {"eval": {"program_html": [{"url": "last"}]}, "intent": "show me X"}
        )

    def test_write_verb_substring(self):
        from lite.gym.envs.browsergym import isolation

        assert isolation.is_mutating({"intent": "Create a new repository named foo"})

    def test_open_a_new_issue_phrasing(self):
        # "open a new issue" slips past the literal "open an issue" verb (669/670).
        from lite.gym.envs.browsergym import isolation

        assert isolation.is_mutating(
            {"intent": "Open a new issue to discuss the implementation of dark mode"}
        )

    def test_promote_to_subreddit(self):
        # reddit submission creation (684-688).
        from lite.gym.envs.browsergym import isolation

        assert isolation.is_mutating(
            {"intent": "Promote byteblaze/dotfiles to subreddit aww with the description"}
        )

    def test_imperative_prefix_rate(self):
        from lite.gym.envs.browsergym import isolation

        assert isolation.is_mutating(
            {"intent": "Rate my recent purchase of floor lamp with 5 stars"}
        )

    def test_noun_ambiguous_verb_as_read(self):
        # "orders"/"review" as nouns mid-sentence must NOT flip a read.
        from lite.gym.envs.browsergym import isolation

        assert not isolation.is_mutating(
            {"intent": "Tell me how many fulfilled orders I have", "eval": {}}
        )

    # --- oracle over real upstream data ---
    def test_oracle_known_writes_and_reads(self):
        """Spot-check the corruption-class false negatives we fixed + a few
        reads that must stay shareable (regression guard, real configs)."""
        pytest.importorskip("webarena", reason="webarena package not installed")
        from lite.gym.envs.browsergym import isolation

        writes = [
            "webarena.669",
            "webarena.670",
            "webarena.684",
            "webarena.688",
            "webarena.585",
            "webarena.792",
        ]
        reads = ["webarena.47", "webarena.704", "webarena.369", "webarena.676"]
        for gid in writes:
            cfg = isolation.task_raw_config("webarena", gid)
            assert cfg is not None and isolation.is_mutating(cfg), f"{gid} must be a writer"
        for gid in reads:
            cfg = isolation.task_raw_config("webarena", gid)
            assert cfg is not None and not isolation.is_mutating(cfg), f"{gid} must stay a reader"

    def test_oracle_csv_requires_reset_all_covered(self):
        """browsergym ships ``requires_reset`` (generated from the same
        upstream json). Every requires_reset=True task MUST classify
        mutating — else the gate would let a backend-dirtying task run as a
        shareable reader. The csv is the oracle; the json is our source."""
        import csv as _csv
        import importlib.resources as ir

        pytest.importorskip("browsergym.experiments", reason="browsergym.experiments not installed")
        for benchmark, fn in [
            ("webarena", "webarena.csv"),
            ("visualwebarena", "visualwebarena.csv"),
        ]:
            pytest.importorskip(benchmark, reason=f"{benchmark} package not installed")
            from lite.gym.envs.browsergym import isolation

            text = ir.files("browsergym.experiments.benchmark.metadata").joinpath(fn).read_text()
            need_reset = [
                r["task_name"]
                for r in _csv.DictReader(io.StringIO(text))
                if r["requires_reset"].strip().lower() == "true"
            ]
            missed = [
                gid
                for gid in need_reset
                if not isolation.is_mutating(isolation.task_raw_config(benchmark, gid) or {})
            ]
            assert not missed, f"{benchmark}: requires_reset tasks classed as readers: {missed}"


# ---------------------------------------------------------------------------
# restore_backend HTTP logic — the WA reset trigger + status-poll (A1/A2)
# ---------------------------------------------------------------------------


class TestFullResetBlocking:
    """``isolation._full_reset_blocking`` — mirrors webarena ``full_reset``:
    GET /reset (tolerate 418) → poll /status until ``"Ready for duty!"`` →
    raise on non-200 / timeout. Mocks ``urllib`` so no reset server is needed
    (the only restore path not already covered by the dispatch-level mocks)."""

    def setup_method(self):
        from lite.gym.envs.browsergym import isolation

        self._orig = (isolation._RESTORE_POLL_INTERVAL_S, isolation._RESTORE_TIMEOUT_S)
        isolation._RESTORE_POLL_INTERVAL_S = 0.01  # tiny — keep tests fast
        isolation._RESTORE_TIMEOUT_S = 0.2  # never-ready case raises in ~0.2s

    def teardown_method(self):
        from lite.gym.envs.browsergym import isolation

        isolation._RESTORE_POLL_INTERVAL_S, isolation._RESTORE_TIMEOUT_S = self._orig

    @staticmethod
    def _resp(status, body=b""):
        class _R:
            def __init__(self):
                self.status = status

            def read(self):
                return body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _R()

    def _patched(self, reset_seq, status_seq):
        """Return a fake urlopen replaying reset_seq for /reset and status_seq
        for /status. Each entry is a ('ok', status, body) tuple or an
        HTTPError code int (urlopen raises HTTPError)."""
        import urllib.error

        st = {"reset": 0, "status": 0}

        def fake(req, timeout=None):
            suffix = "reset" if req.full_url.endswith("/reset") else "status"
            seq = reset_seq if suffix == "reset" else status_seq
            i = min(st[suffix], len(seq) - 1)
            st[suffix] += 1
            entry = seq[i]
            if isinstance(entry, int):  # HTTPError code
                raise urllib.error.HTTPError(req.full_url, entry, "err", {}, None)
            _, code, body = entry
            return self._resp(code, body)

        return fake

    def test_reset_then_status_becomes_ready(self):
        from lite.gym.envs.browsergym import isolation

        fake = self._patched(
            reset_seq=[("ok", 200, b"Reset started.")],
            status_seq=[("ok", 200, b"running"), ("ok", 200, b"Ready for duty!")],
        )
        with patch("urllib.request.urlopen", fake):
            isolation._full_reset_blocking("http://reset", "webarena")  # no raise

    def test_reset_418_already_running_is_tolerated(self):
        from lite.gym.envs.browsergym import isolation

        fake = self._patched(
            reset_seq=[418],  # already running → tolerated
            status_seq=[("ok", 200, b"Ready for duty!")],
        )
        with patch("urllib.request.urlopen", fake):
            isolation._full_reset_blocking("http://reset", "webarena")  # no raise

    def test_reset_500_raises(self):
        from lite.gym.envs.browsergym import isolation

        fake = self._patched(reset_seq=[500], status_seq=[("ok", 200, b"Ready for duty!")])
        with patch("urllib.request.urlopen", fake):
            with pytest.raises(RuntimeError):
                isolation._full_reset_blocking("http://reset", "webarena")

    def test_status_non_200_raises(self):
        from lite.gym.envs.browsergym import isolation

        fake = self._patched(
            reset_seq=[("ok", 200, b"")], status_seq=[("ok", 503, b"oops")]
        )  # non-200 status → fatal
        with patch("urllib.request.urlopen", fake):
            with pytest.raises(RuntimeError):
                isolation._full_reset_blocking("http://reset", "webarena")

    def test_status_never_ready_times_out(self):
        from lite.gym.envs.browsergym import isolation

        fake = self._patched(
            reset_seq=[("ok", 200, b"")], status_seq=[("ok", 200, b"still running")]
        )  # never "Ready for duty!"
        with patch("urllib.request.urlopen", fake):
            with pytest.raises(RuntimeError):
                isolation._full_reset_blocking("http://reset", "webarena")


# ---------------------------------------------------------------------------
# Env-var checks
# ---------------------------------------------------------------------------


class TestCheckEnvVars:
    def setup_method(self):
        # Snapshot env to restore after each test (so we don't poison sibling tests).
        self._orig = {k: os.environ.get(k) for d in _DEFAULT_ENV_VARS.values() for k in d}

    def teardown_method(self):
        for k, v in self._orig.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_miniwob_sets_default_url(self):
        # Default is the shared-singleton preferred port 7560 (NOT 8080, which
        # collides with webgym's OmniBoxes node — see _ensure_miniwob_singleton).
        os.environ.pop("MINIWOB_URL", None)
        _check_env_vars("miniwob")
        assert os.environ.get("MINIWOB_URL") == "http://localhost:7560/miniwob/"

    def test_existing_value_not_overridden(self):
        os.environ["MINIWOB_URL"] = "http://custom:9000/"
        _check_env_vars("miniwob")
        assert os.environ["MINIWOB_URL"] == "http://custom:9000/"

    def test_unknown_benchmark_noop(self):
        # No defaults / no requireds → does not raise.
        _check_env_vars("nonsense_bench")


class TestMiniwobSharedSingleton:
    """miniwob is a host-wide SHARED singleton with conflict-avoidance: a flock'd
    registry makes every env-server converge on ONE instance (no fan-out leak),
    while the port is still auto-picked if the preferred one is busy."""

    def _isolate(self, monkeypatch, tmp_path):
        from lite.gym.envs.browsergym import main as m

        reg, lock = tmp_path / "miniwob-singleton.json", tmp_path / "miniwob.lock"
        monkeypatch.setattr(m, "_miniwob_registry_paths", lambda: (reg, lock))
        for v in ("MINIWOB_PORT", "MINIWOB_URL"):
            monkeypatch.delenv(v, raising=False)
        return m, reg

    def test_operator_override_pins_port_no_registry(self, monkeypatch, tmp_path):
        m, reg = self._isolate(monkeypatch, tmp_path)
        monkeypatch.setenv("MINIWOB_PORT", "9091")
        monkeypatch.setattr(m, "_service_up", lambda b: True)  # pretend already up
        m._ensure_miniwob_singleton()
        assert os.environ["MINIWOB_URL"] == "http://localhost:9091/miniwob/"
        assert not reg.exists()  # operator-pinned → never touches the shared registry

    def test_reuse_live_registered_singleton_no_start(self, monkeypatch, tmp_path):
        m, reg = self._isolate(monkeypatch, tmp_path)
        m._write_miniwob_registry(7575)
        monkeypatch.setattr(m, "_service_up", lambda b: True)  # registered one is alive
        started = []
        monkeypatch.setattr(m, "_run_miniwob_start_sh", lambda: started.append(1))
        m._ensure_miniwob_singleton()
        assert os.environ["MINIWOB_PORT"] == "7575"  # reused the registered port
        assert started == []  # did NOT start a second server

    def test_pick_and_register_when_none_live(self, monkeypatch, tmp_path):
        m, reg = self._isolate(monkeypatch, tmp_path)
        # Nothing registered; preferred port "free"; start brings it up.
        from lite.gym.utils.backend import ports as _port

        monkeypatch.setattr(_port, "_is_port_free", lambda p: True)
        up = {"v": False}
        monkeypatch.setattr(m, "_service_up", lambda b: up["v"])
        monkeypatch.setattr(m, "_miniwob_alive_on", lambda p: False)

        def _fake_start():
            up["v"] = True

        monkeypatch.setattr(m, "_run_miniwob_start_sh", _fake_start)
        m._ensure_miniwob_singleton()
        assert os.environ["MINIWOB_PORT"] == str(m._MINIWOB_PORT_DEFAULT)
        assert m._read_miniwob_registry() == m._MINIWOB_PORT_DEFAULT  # recorded for reuse

    def test_pick_avoids_busy_preferred_port(self, monkeypatch, tmp_path):
        m, _ = self._isolate(monkeypatch, tmp_path)
        from lite.gym.utils.backend import ports as _port

        lo, hi = m._MINIWOB_PORT_RANGE
        # Preferred busy + not-a-miniwob; only lo+1 is free in the range.
        monkeypatch.setattr(m, "_miniwob_alive_on", lambda p: False)
        monkeypatch.setattr(_port, "_is_port_free", lambda p: p == lo + 1)
        assert m._pick_miniwob_port() == lo + 1

    def test_pick_adopts_live_miniwob_on_preferred(self, monkeypatch, tmp_path):
        m, _ = self._isolate(monkeypatch, tmp_path)
        from lite.gym.utils.backend import ports as _port

        # Preferred port is "in use" but it IS a live miniwob → adopt, don't skip.
        monkeypatch.setattr(m, "_miniwob_alive_on", lambda p: p == m._MINIWOB_PORT_DEFAULT)
        monkeypatch.setattr(_port, "_is_port_free", lambda p: False)
        assert m._pick_miniwob_port() == m._MINIWOB_PORT_DEFAULT

    @pytest.mark.live
    def test_shutdown_and_reap_never_kill_miniwob(self, monkeypatch):
        """Cross-kill regression: a co-resident env-server's shutdown / boot-reap
        must NOT kill the host-shared miniwob (it has no per-server owner; killing
        it would tear down ANOTHER live server's backend mid-episode). Stands up a
        dummy http.server standing in for the shared miniwob and asserts it
        survives both lifecycle hooks."""
        import socket
        import subprocess
        import time

        from lite.gym.envs.browsergym import main as m
        from lite.gym.remote.scope import ServerScope

        # WA teardown is orthogonal here — stub it so the test needs no docker.
        monkeypatch.setattr(m, "_wa_stop_scoped", lambda sp: 0)
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        proc = subprocess.Popen(
            ["python3", "-m", "http.server", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(1.0)
            assert proc.poll() is None, "dummy miniwob failed to start"
            monkeypatch.setenv("MINIWOB_PORT", str(port))
            scope = ServerScope.from_server(server_port=30111, strict_token=None)
            svc = m.BrowserGymServices()
            svc.shutdown("browsergym", scope)
            svc.reap("browsergym", scope, set(), boot=True)
            time.sleep(0.5)
            assert proc.poll() is None, (
                "shutdown/reap killed the shared miniwob — cross-kill regression!"
            )
        finally:
            proc.kill()
            proc.wait()


# ---------------------------------------------------------------------------
# ensure_services: two-cache eviction (I2) + homepage gate wiring (I3)
# ---------------------------------------------------------------------------


class TestEvictServicesCache:
    """I2: a WA/VWA service that DIED mid-run is short-circuited by TWO
    caches — browsergym main's ``_services_started`` AND the registry-level
    ensure cache (which runs FIRST). ``_evict_services_cache`` must clear
    BOTH; clearing only the local one is dead code (the registry cache never
    lets ``ensure`` re-enter)."""

    def test_evict_clears_both_caches(self):
        import sys

        from lite.gym.envs.browsergym import main as m

        registry_mod = sys.modules["lite.gym.registry"]
        m._services_started.add("webarena")
        registry_mod._services_started.add("browsergym.webarena")
        try:
            m._evict_services_cache("webarena")
            assert "webarena" not in m._services_started
            assert "browsergym.webarena" not in registry_mod._services_started, (
                "registry-level ensure cache not evicted — the local-only "
                "eviction is dead code (registry short-circuits first)"
            )
        finally:
            m._services_started.discard("webarena")
            registry_mod._services_started.discard("browsergym.webarena")


class TestEnsureServicesHomepageGate:
    """I3: ``_ensure_services`` must gate WA/VWA on ``_homepage_up`` — the
    homepage Flask is launched LAST by start.sh, so 'shared sites up but
    homepage cold' must stay a retriable 503 (not a silent early-return that
    marks the benchmark started with the homepage never launched). Fully
    hermetic: probes + start.sh are stubbed, no docker."""

    def test_homepage_down_is_retriable_503_until_up(self, monkeypatch):
        import subprocess as _subprocess
        from types import SimpleNamespace

        from lite.gym.envs.browsergym import main as m
        from lite.gym.errors import CapacityExhausted

        m._services_started.discard("webarena")
        monkeypatch.setattr(m, "_auto_pick_webarena_ports", lambda: None)
        monkeypatch.setattr(m, "_service_up", lambda b: True)
        monkeypatch.setattr(m, "_classifieds_up", lambda: True)
        monkeypatch.setattr(m, "_wa_down_sites", lambda b: [])
        homepage = {"up": False}
        monkeypatch.setattr(m, "_homepage_up", lambda: homepage["up"])
        started: list = []
        monkeypatch.setattr(
            m,
            "subprocess",
            SimpleNamespace(
                run=lambda *a, **k: (
                    started.append(a) or SimpleNamespace(returncode=0, stdout="", stderr="")
                ),
                TimeoutExpired=_subprocess.TimeoutExpired,
                CalledProcessError=_subprocess.CalledProcessError,
            ),
        )
        try:
            with pytest.raises(CapacityExhausted) as exc:
                m._ensure_services("browsergym.webarena")
            assert "homepage" in str(exc.value), "the warming 503 must name the cold homepage"
            assert started, "the idempotent start.sh must have been (re-)entered"
            assert "webarena" not in m._services_started, (
                "a homepage-cold benchmark must never be cached as started"
            )
            # Homepage comes up → same call now passes the early gate: no
            # start.sh re-entry, benchmark cached as started, no raise.
            homepage["up"] = True
            started.clear()
            m._ensure_services("browsergym.webarena")
            assert "webarena" in m._services_started
            assert not started
        finally:
            m._services_started.discard("webarena")


def test_map_is_setup_singleton_not_webarena_readiness_gate(monkeypatch):
    from lite.gym.envs.browsergym import main as m

    ready_site_names = {row[0] for row in m._WA_SITE_SPECS}
    assert "map" not in ready_site_names

    map_rows = [row for row in m._WA_PORT_PLAN if row["port_var"] == "MAP_PORT"]
    assert len(map_rows) == 1
    assert map_rows[0]["singleton"] is True
    assert map_rows[0]["url_vars"] == {"WA_MAP": "http://localhost:{port}"}

    monkeypatch.setattr(m, "_WA_PORT_PLAN", [dict(map_rows[0])])
    monkeypatch.delenv("MAP_PORT", raising=False)
    monkeypatch.delenv("WA_MAP", raising=False)
    m._auto_pick_webarena_ports()
    assert os.environ["MAP_PORT"] == str(map_rows[0]["default"])
    assert os.environ["WA_MAP"] == f"http://localhost:{map_rows[0]['default']}"


def test_browsergym_readme_documents_cache_and_start_exports():
    root = Path(__file__).resolve().parents[4]
    text = (root / "lite" / "gym" / "envs" / "browsergym" / "README.md").read_text(encoding="utf-8")
    for needle in [
        "BROWSERGYM_CACHE",
        "missing `.cache/` in the current checkout",
        "is not an asset blocker",
        "source lite/gym/envs/browsergym/scripts/start.sh miniwob",
        "source lite/gym/envs/browsergym/scripts/start.sh visualwebarena",
        "MINIWOB_URL",
        "WA_*",
        "VWA_*",
        "VWA_CLASSIFIEDS_RESET_TOKEN",
        "OPENAI_API_KEY",
        "CUA_LITE_ENV_SERVER_TOKEN",
        "WA_MAP",
        "WA_HOMEPAGE",
        "VWA_HOMEPAGE",
        "BROWSERGYM_START_MAP=0",
    ]:
        assert needle in text
