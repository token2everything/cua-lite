"""Shared-backend isolation metadata for WebArena-family benchmarks.

WebArena, WebArena-Verified, and VisualWebArena each back tasks with a shared,
mutable site stack. The env-server's generic conflict gate
(:mod:`lite.gym.remote.conflict`) serializes writers per stack by reading
``metadata.others["conflict_keys"]`` + ``metadata.others["mutating"]`` (an
opt-in, server-mode-only contract carried in ``others`` rather than typed
core fields). This module derives those two values at registration time
(``conflict_keys_and_mutating``) and provides the ``restore_backend`` hook
the server dispatches after a writer closes.

Kept separate from ``main.py`` (the env wrapper) so the isolation concern
is one discrete, swappable unit — symmetric with the server's
``conflict.py``. The server stays env-agnostic; all WA-specific policy
(sites → coarse key, write classification, how to reset the stack) lives
here.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

from lite.gym.utils import config as env_config

logger = logging.getLogger(__name__)

# Registration defaults — see configs/default.yaml (override the WHOLE config via
# BROWSERGYM_CONFIG=isolation, or =/path/to.yaml). The yaml is the single source
# for the isolation server_kwargs read below. ENV_DIR is the browsergym PACKAGE
# dir (this module sits directly in it) where configs/ lives.
ENV_DIR = str(Path(__file__).parent)
CFG = env_config.load(ENV_DIR)

# ============================================================================
# Config defaults — every value below is read once from configs/default.yaml
# via env_config.load(ENV_DIR). Swap the whole file at startup with
# BROWSERGYM_CONFIG=<abs-path | bundled-name> (e.g. BROWSERGYM_CONFIG=isolation
# selects the bundled configs/isolation.yaml). These are deployment defaults.
# ============================================================================
# --- server_kwargs (per-deployment) ---
_BACKEND_ISOLATION       = CFG.server_kwargs["backend_isolation"]
# Full-reset poll cadence — mirrors webarena's ``full_reset`` (a WA stack reset
# takes 200-500 s): poll ``/status`` every interval until ready, cap the wait.
# The tests mutate these module globals directly. (yaml already types them.)
_RESTORE_POLL_INTERVAL_S = CFG.server_kwargs["restore_poll_s"]
_RESTORE_TIMEOUT_S       = CFG.server_kwargs["restore_timeout_s"]
# L2 dead-backend fail-fast threshold (see configs/default.yaml). Consumed by
# main.py's step loop; 0 disables.
_POOL_UNREACHABLE_FAILED_STEPS = CFG.server_kwargs["pool_unreachable_failed_steps"]
# ============================================================================

# Per-benchmark {bgym_task_id -> raw task config}, loaded once from the
# upstream ``test.raw.json`` and cached. Single source for exclude_reason
# (main.py) + conflict_keys + mutating — one json load, not three.
_RAW_CONFIG_CACHE: dict[str, dict[str, dict]] = {}


def task_raw_config(benchmark: str, bgym_task_id: str) -> dict | None:
    """Raw upstream task config for a WA/VWA task, or ``None`` (miniwob,
    unknown id). Cached per benchmark.

    Loaded straight from the upstream package's bundled config via
    ``importlib.resources`` — same source browsergym itself reads (its
    ``benchmark.csv`` is *generated* from this json, so this is the
    authoritative superset, carrying the ``intent`` + ``program_html`` url
    the csv drops). The package is a hard dep of any WA/VWA task, so the
    resource is guaranteed present whenever this is reached."""
    if benchmark not in ("webarena", "webarena_verified", "visualwebarena"):
        return None
    if benchmark not in _RAW_CONFIG_CACHE:
        if benchmark == "webarena_verified":
            from lite.gym.envs.browsergym import webarena_verified

            _RAW_CONFIG_CACHE[benchmark] = {
                task_id: record
                for task_id, record in zip(
                    webarena_verified.task_ids(),
                    webarena_verified.task_records(),
                    strict=True,
                )
            }
        else:
            import importlib.resources as ir
            import json

            pkg, fn = (("webarena", "test.raw.json") if benchmark == "webarena"
                       else ("visualwebarena", "test_raw.json"))
            tasks = json.loads(ir.files(pkg).joinpath(fn).read_text())
            _RAW_CONFIG_CACHE[benchmark] = {
                f"{benchmark}.{t['task_id']}": t for t in tasks
            }
    return _RAW_CONFIG_CACHE[benchmark].get(bgym_task_id)


_DEPENDS_ON_CACHE: dict[str, dict[str, list[str]]] = {}


def task_depends_on(benchmark: str, bgym_task_id: str) -> list[str]:
    """The hand-curated dependency parents of a WA/VWA task — BrowserGym's
    ``depends_on`` metadata column. It is NOT a data-flow prerequisite (it isn't
    in the raw ``test.raw.json``; WA's ``require_reset`` is all-False): it's a
    conservative *run order* so an earlier writer's residue can't false-satisfy a
    later task. Running the suite in this topological order needs only ONE full
    backend reset at the start (BrowserGym's ``dependency_graphs_over_env_args``
    model) — vs a reset after every writer. So the win is the reset COUNT (one per
    suite, not one per writer); the documented strict flow runs the write pass
    serially in this order (docs/eval.md).

    Loaded from BrowserGym's generated ``experiments`` metadata CSV (the only
    place ``depends_on`` lives). Empty for miniwob / unknown id / if that optional
    metadata package isn't installed (→ no curated ordering; writers fall back to
    plain registration/task-id order)."""
    if benchmark not in ("webarena", "webarena_verified", "visualwebarena"):
        return []
    if benchmark not in _DEPENDS_ON_CACHE:
        table: dict[str, list[str]] = {}
        try:
            import csv
            import importlib.util
            import io

            # Locate the namespace package without importing
            # ``browsergym.experiments``: its ``__init__`` imports pandas even
            # though this path needs only the packaged static CSV.
            spec = importlib.util.find_spec("browsergym")
            if spec is None or not spec.submodule_search_locations:
                raise FileNotFoundError("browsergym package not installed")
            csv_path = (
                Path(next(iter(spec.submodule_search_locations)))
                / "experiments"
                / "benchmark"
                / "metadata"
                / f"{benchmark}.csv"
            )
            txt = csv_path.read_text()
            for row in csv.DictReader(io.StringIO(txt)):
                table[row["task_name"]] = (row.get("depends_on") or "").split()
        except Exception as e:  # noqa: BLE001 — optional metadata; degrade to no ordering
            logger.warning(
                "task_depends_on(%s): dependency metadata unavailable (%s) — "
                "no topological ordering; writers fall back to registration order.",
                benchmark, e,
            )
        _DEPENDS_ON_CACHE[benchmark] = table
    return _DEPENDS_ON_CACHE[benchmark].get(bgym_task_id, [])


# Conservative write detection: a task is ``mutating`` if ANY signal fires
# (correctness > parallelism — over-marking only costs read
# concurrency, never correctness). (1) WebArena's own ``require_reset``
# (real in VWA, all-False in WA); (2) a re-navigation eval that re-fetches a
# SPECIFIC url (not ``"last"``) to verify persisted state — ``program_html``
# or its visual analog ``page_image_query``; (3) a write verb in the intent
# (substring list below, or a noun-ambiguous verb as a sentence-leading
# imperative — see ``_WRITE_VERB_PREFIXES``).
_WRITE_VERBS = (
    "post ", "add ", "delete", "change", "edit", "make ", "submit",
    "reply", "assign", "draft", "send ", "upvote", "downvote", "fork",
    "rename", "disable", "enable", "reject", "open an issue", "close ",
    "subscribe", "unsubscribe", "like ", "star ", "move ", "upload",
    # "open a new issue" (gitlab 669/670) slips past "open an issue";
    # "promote X to subreddit Y" (reddit 684-688) creates a submission;
    # "notify X in their order" (shopping_admin 491; 492-495 already caught)
    # adds an admin order-comment → a write the agent attempts regardless of
    # the eval expecting "N/A".
    "open a new issue", "promote ", "notify ",
)
# These verbs need WORD-BOUNDARY matching: as bare substrings they matched
# nouns ("cancelled", "Approved reviews", "latest updated/created issue",
# "makeup remover", "asset/headset", "comments section") and needlessly
# serialized ~31 read tasks. ``\bverb\b`` still catches the real writes
# ("cancel order", "create a repo", "set status", "add a comment") — strictly
# tighter, so it drops only the noun matches (audit-verified: 0 require_reset
# or eval-signal writes lost). Frees 24 WA + 7 VWA reads (North Star #2).
_WRITE_VERBS_WORD = ("cancel", "approve", "update", "create", "remove", "set", "comment")
_WRITE_WORD_RE = re.compile(r"\b(?:" + "|".join(_WRITE_VERBS_WORD) + r")\b")
# Noun-ambiguous verbs (rate/buy/order/review/book...) appear far more often
# as nouns in READ intents ("how many *orders*", "the cheapest *book*",
# "what size should I *buy*") than as writes, so substring-matching them
# would wrongly serialize ~100 reads. The genuine writes are imperative —
# the verb leads the sentence ("Rate my recent purchase...", "Buy the
# highest-rated product...", "Purchase the exact item...") — so we match
# these only as a sentence-leading prefix. On the WA/VWA suites this flips
# exactly the 16 real purchase/rating writes and zero noun-reads.
_WRITE_VERB_PREFIXES = (
    "rate ", "buy ", "purchase ", "order ", "book ", "reserve ",
    "rent ", "checkout ", "review ", "leave a ",
)
_STATIC_SITES = {"wikipedia", "map"}  # frozen dumps → physically read-only


def is_mutating(config: dict) -> bool:
    if config.get("require_reset"):
        return True
    eval_ = config.get("eval", {}) or {}
    for ph in eval_.get("program_html", []) or []:
        if ph.get("url") not in ("last", "", None):
            return True
    # Visual analog of the program_html re-navigation check (VWA): a
    # ``page_image_query`` whose ``eval_image_url`` re-fetches a SPECIFIC
    # url (not ``"last"``) verifies persisted state → a write. Today every
    # such task is already caught by another signal (0 net), but this keeps
    # the eval-based detection symmetric so a future visual-only write
    # can't slip through.
    for pq in eval_.get("page_image_query", []) or []:
        if pq.get("eval_image_url") not in ("last", "", None):
            return True
    intent = (config.get("intent") or "").lower()
    if any(v in intent for v in _WRITE_VERBS):
        return True
    if _WRITE_WORD_RE.search(intent):
        return True
    lead = intent.lstrip()
    return any(lead.startswith(p) for p in _WRITE_VERB_PREFIXES)


def conflict_keys_and_mutating(
    benchmark: str, bgym_task_id: str
) -> tuple[tuple[str, ...], bool]:
    """Registration-time (conflict_keys, mutating) for a browsergym task.

    Isolation is **OFF by default** — defaults are permissive/fast; rigor
    is opt-in (consistent with the other rigor knobs: reset endpoints and
    site URLs are also operator-set). The naive rollout runs fully
    concurrent (the prior behavior), so isolation-on doesn't surprise an
    operator with serialized writers + 503 retries that the default
    ``--max-attempts`` can't absorb. Launch with ``BROWSERGYM_CONFIG=isolation``
    (the bundled ``configs/isolation.yaml``, which sets
    ``server_kwargs.backend_isolation: "strict"`` — the only value that
    engages the gate); pair it with a ``*_FULL_RESET`` endpoint + a raised
    ``--max-attempts`` for full eval rigor.

    When engaged it emits ONE COARSE key for the whole shared stack (WA and
    VWA share containers → same id; per-site keying is unsafe under agent
    wandering) only for WA/VWA *mutable-site* tasks — miniwob,
    static-only (wikipedia/map), and every non-browsergym env get ``()``
    (gate no-op, nothing to protect). When the gate IS engaged its
    behavior never trades correctness for speed (North Star #1)."""
    if _BACKEND_ISOLATION != "strict":
        return (), False
    config = task_raw_config(benchmark, bgym_task_id)
    if config is None:
        return (), False
    sites = config.get("sites") or []
    if not any(s not in _STATIC_SITES for s in sites):
        return (), False  # static-only → read-safe, no key
    if benchmark == "webarena_verified":
        from lite.gym.envs.browsergym import webarena_verified

        mutating = webarena_verified.is_mutating(bgym_task_id)
    else:
        mutating = is_mutating(config)
    return ("webarena",), mutating


async def restore_backend(env_id: str, key: str) -> None:
    """Server-dispatched hook: reset the shared WA
    stack ``key`` to a clean baseline after a writer episode closes. The
    server holds the conflict key across this call and releases it **only
    if we return normally**, so the next claimant never sees a half-reset
    stack (and, if we raise, never sees a *dirty* one — correctness).

    Mirrors ``browsergym.webarena.instance.full_reset``: ``GET
    $WA_FULL_RESET/reset`` (tolerating 418 "already running"), then poll
    ``/status`` until the body is ``"Ready for duty!"``. The blocking HTTP
    + sleep loop runs in a worker thread so the 200-500 s reset never
    freezes the server event loop (``_close_quietly`` is detached but still
    on the loop).

    OPT-IN: the reset endpoint is read from ``$WA_FULL_RESET`` /
    ``$VWA_FULL_RESET`` (exported by ``scripts/start.sh``'s
    ``export_webarena_env``, empty unless the operator sets it). This is the
    ONLY mechanism that clears cross-task DB residue on the WRITABLE benches
    (shopping / shopping_admin / gitlab / postmill) — upstream provides no
    cheaper per-task reseed (even VWA's ``reset_shopping.sh`` is a full
    ``docker stop/rm/run`` of the populated image, ~1-2 min/site; the
    canonical WA reset is per-suite, "after evaluating the 812 examples").
    Because that cost is a throughput killer it stays DISABLED by default;
    when no endpoint is set this logs + no-ops, and INTERLEAVING is still
    prevented by the conflict gate (only cross-episode RESIDUE cleanup is
    skipped). A configured-but-failing reset **raises**, so the caller keeps
    the key held rather than re-leasing a dirty backend."""
    reset_base = os.environ.get("WA_FULL_RESET") or os.environ.get("VWA_FULL_RESET")
    if not reset_base:
        logger.warning(
            "restore_backend(%s, %s): no WA_FULL_RESET / VWA_FULL_RESET endpoint "
            "configured — skipping cross-task residue reset on the shared "
            "writable backend (interleaving still prevented by the conflict "
            "gate; set WA_FULL_RESET to a full-reset service to enable, see "
            "scripts/start.sh export_webarena_env).",
            env_id, key,
        )
        return
    import asyncio
    await asyncio.to_thread(_full_reset_blocking, reset_base.rstrip("/"), key)


def _full_reset_blocking(base: str, key: str) -> None:
    """Trigger + await one WA full-reset (blocking; run via ``to_thread``).
    Raises ``RuntimeError`` on any hard failure or timeout."""
    import urllib.error as ue
    import urllib.request as ur
    # 1) trigger the reset (GET, like the reference); 418 == already running.
    try:
        with ur.urlopen(ur.Request(f"{base}/reset"), timeout=60) as resp:
            logger.info("restore_backend(%s): reset started (%s)", key, resp.status)
    except ue.HTTPError as e:
        if e.code != 418:
            raise RuntimeError(f"restore_backend({key}) reset trigger failed: {e}") from e
        logger.warning("restore_backend(%s): reset already running, awaiting status", key)
    # 2) poll /status until ready — the conflict key stays held meanwhile.
    # Wall-clock deadline (like the reference's time.time()-start) — robust
    # even if the poll interval is 0, and counts the urlopen time, not just
    # the sleeps.
    start = time.monotonic()
    while True:
        with ur.urlopen(ur.Request(f"{base}/status"), timeout=60) as resp:
            if resp.status != 200:  # reference treats non-200 status as fatal
                raise RuntimeError(
                    f"restore_backend({key}) status request failed ({resp.status})")
            body = resp.read().decode("utf-8", "replace").strip()
        if body == "Ready for duty!":
            logger.info("restore_backend(%s): stack ready after %.0fs",
                        key, time.monotonic() - start)
            return
        elapsed = time.monotonic() - start
        if elapsed >= _RESTORE_TIMEOUT_S:
            raise RuntimeError(
                f"restore_backend({key}) not ready after {elapsed:.0f}s "
                f"(> {_RESTORE_TIMEOUT_S:.0f}s); status={body!r}")
        time.sleep(_RESTORE_POLL_INTERVAL_S)
