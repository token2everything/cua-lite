"""Three-layer admission for the env-server.

* L1 host sensors — RAM / load / disk / swap via psutil; pluggable.
* L2 in-flight cap — auto-derived from vCPU AND RAM; ``--max-live-envs`` overrides.
* L3 env-internal — each env raises ``CapacityExhausted`` from its own
  ``reset()`` when its bounded resource is full.

Docker semaphore is structural (not admission), serialized at the
process level so sync container factories and async sandbox/base.py share
one daemon slot count. Its capacity derives from host vCPU + storage
class (NVMe vs HDD) — disk I/O is the actual bottleneck.

It exists ONLY under an env-server. A process that never builds an
:class:`AdmissionGate` — direct rollout, the oracle validators — leaves
``_docker_sema`` unset, and both slot helpers then yield unpaced. Direct mode
has no 503-retry client, so a slot timeout there is a hard failure rather than
back-pressure; a direct-mode caller bounds its own fan-out.

All defaults are sized from :class:`HostCapacity` (single source of
truth in :mod:`lite.gym.utils.server.capacity`); hardcoded numbers in
this module are conservative tests-only defaults on
:class:`AdmissionConfig` so test fixtures stay terse.

This module is env-agnostic. Operator-facing CLI lives in
``scripts/serve_env.py``; env-var escape hatches for this config are read here
so the launcher can stay focused on startup flow.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import threading
import time
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass

import psutil

from lite.gym.errors import CapacityExhausted
from lite.gym.utils.server.capacity import (
    HostCapacity,
    cached_host_capacity,
    effective_cpu_count,
)

# Re-export so existing callers (``from lite.gym.remote.admission import
# effective_cpu_count``) keep working — the function lives in
# :mod:`lite.gym.utils.server.capacity` now. New code should prefer the
# ``capacity`` source directly.
__all__ = [
    "AdmissionConfig",
    "AdmissionGate",
    "CapacityExhausted",
    "DOCKER_CREATE_WAIT_TIMEOUT_S",
    "EMERGENCY_RETRY_AFTER_S",
    "configure_docker_sema",
    "derive_admission_config",
    "derive_docker_create_concurrency",
    "derive_max_live_envs",
    "docker_create_slot",
    "docker_create_slot_async",
    "effective_cpu_count",
    "log_admission_config",
    "register_emergency_check",
]

logger = logging.getLogger(__name__)


#: Floor for ``Retry-After`` hints on any admission 503. Adaptive
#: docker_sema retry-after at :func:`_docker_sema_retry_after_s`
#: clamps no lower than this; L1/L2 emergency raises use it directly.
#: Sized to match the client's minimum sleep granularity.
EMERGENCY_RETRY_AFTER_S: int = 10

#: Bounded wait inside docker-create slot before raising. Caps daemon stall.
DOCKER_CREATE_WAIT_TIMEOUT_S: float = 5.0


# --- Config (tier-2: env vars / CLI flag) ---

@dataclass(frozen=True)
class AdmissionConfig:
    """Per-process admission config.

    Field defaults are the conservative-minimum acceptable values for
    test fixtures — production code path uses
    :func:`derive_admission_config` so the production defaults derive
    from :class:`HostCapacity` rather than landing here as magic
    numbers.

    ``emergency_disk_free_min_bytes=0`` and ``emergency_swap_pct=100.0``
    are sentinel "disabled" values — keeps the L1 checks no-op for
    legacy fixtures that don't set them. Production gets non-zero
    floors via :func:`derive_admission_config`.
    """
    max_live_envs: int
    emergency_ram_pct: float = 95.0
    emergency_ram_free_min_bytes: int = 2 * (1024 ** 3)
    emergency_load_per_cpu: float = 2.5
    docker_create_concurrency: int = 8
    emergency_disk_free_min_bytes: int = 0    # 0 == disabled
    emergency_swap_pct: float = 100.0          # 100 == disabled


# --- Cluster auto-derivation ---

def derive_max_live_envs(
    *,
    ram_gb: float,
    vcpu: int | None = None,
    per_vcpu: int | None = None,
    floor: int | None = None,
    ceiling: int | None = None,
) -> int:
    """L2 cap from MIN(vCPU-bound, RAM-bound).

    The cap is the smaller of two budgets:

    * **vCPU-bound**: ``vcpu * per_vcpu`` (default ``per_vcpu=4``)
    * **RAM-bound**: ``ram_gb / per_env_ram_gb`` (default 4 via
      ``CUA_LITE_MAX_LIVE_PER_ENV_RAM_GB`` — reserves ~2 GB for the env itself plus
      headroom for the docker daemon, training process, OS page cache,
      and other co-resident workloads)

    Defaults are biased toward **safety in multi-tenant + leak-bounded
    scenarios** rather than maximum throughput on a single-tenant host:

    * ``CUA_LITE_MAX_LIVE_PER_ENV_RAM_GB=4`` (was 2) — the original "2 GB/env" matched a
      single lite.osworld container's average RSS but ignored the
      RAM other co-resident processes need. 4 GB/env keeps the cap
      below the "if every client dies it leaves N containers leaked
      for up to ``CUA_LITE_WARM_SESSION_TIMEOUT_S``" line.
    * ``CUA_LITE_MAX_LIVE_CEILING=512`` (was 4096) — the previous
      4096 was effectively no ceiling on any realistic host. 512 is a
      meaningful cap that aligns with "worst-case container leak on
      session_timeout is bounded by something operators can reason
      about." Single-tenant high-throughput callers should override
      explicitly via ``--max-live-envs`` or this env var.

    Overrides via ``CUA_LITE_MAX_LIVE_PER_VCPU`` (default 4),
    ``CUA_LITE_MAX_LIVE_PER_ENV_RAM_GB`` (default 4),
    ``CUA_LITE_MAX_LIVE_FLOOR`` (16), ``CUA_LITE_MAX_LIVE_CEILING`` (512).
    """
    vcpu_n: int = vcpu if vcpu is not None else cached_host_capacity().vcpu
    pv: int = (
        per_vcpu if per_vcpu is not None
        else int(os.environ.get("CUA_LITE_MAX_LIVE_PER_VCPU", "4"))
    )
    # Floor 4 (was 16): a 4-vCPU/16 GB laptop's RAM bound is 4 envs;
    # forcing 16 there would oversubscribe RAM into OOM. 4 still lets
    # the smallest host run *something*; operators wanting more pick a
    # bigger box.
    floor_n: int = (
        floor if floor is not None
        else int(os.environ.get("CUA_LITE_MAX_LIVE_FLOOR", "4"))
    )
    ceiling_n: int = (
        ceiling if ceiling is not None
        else int(os.environ.get("CUA_LITE_MAX_LIVE_CEILING", "512"))
    )
    vcpu_bound = vcpu_n * pv
    per_env_ram: float = float(
        os.environ.get("CUA_LITE_MAX_LIVE_PER_ENV_RAM_GB", "4")
    )
    ram_bound = int(ram_gb / per_env_ram) if per_env_ram > 0 else vcpu_bound
    bound = min(vcpu_bound, ram_bound)
    return max(min(bound, ceiling_n), floor_n)


def derive_docker_create_concurrency(
    *,
    host: HostCapacity | None = None,
) -> int:
    """Structural docker-create rate limit derived from host capacity.

    PER-PROCESS, and the distinction is load-bearing: N processes each honouring
    this number put N× this many creates on one daemon. It is the right bound for
    the shipped configurations — the env-server is a single uvicorn process
    (``scripts/serve_env.py`` passes no ``workers=``) and every shipped
    ``devs/exps/eval/*/run.sh`` ``exec``s exactly one ``scripts/rollout.py`` — so
    "one process" and "one host" coincide there. Running several validators by
    hand does NOT, and nothing coordinates them; bound the fan-out at the
    launcher instead.

    Image extraction is disk-I/O bound, so the ratio depends on storage
    class:

    * NVMe/SSD: ``vcpu * 0.25`` (default) — fast random write sustains
      higher daemon parallelism
    * HDD: ``vcpu * 0.0625`` — head seeks serialize the work

    Floor 8 protects small hosts; ceiling 64 prevents overwhelming the
    docker daemon's internal goroutine pool.

    Overrides via ``CUA_LITE_DOCKER_CREATE_PER_VCPU_NONROTATIONAL`` (0.25),
    ``CUA_LITE_DOCKER_CREATE_PER_VCPU_ROTATIONAL`` (0.0625),
    ``CUA_LITE_DOCKER_CREATE_FLOOR`` (8),
    ``CUA_LITE_DOCKER_CREATE_CEILING`` (64).
    """
    host_r: HostCapacity = host if host is not None else cached_host_capacity()
    nvme: float = float(
        os.environ.get("CUA_LITE_DOCKER_CREATE_PER_VCPU_NONROTATIONAL", "0.25")
    )
    hdd: float = float(
        os.environ.get("CUA_LITE_DOCKER_CREATE_PER_VCPU_ROTATIONAL", "0.0625")
    )
    floor_n: int = int(os.environ.get("CUA_LITE_DOCKER_CREATE_FLOOR", "8"))
    ceiling_n: int = int(os.environ.get("CUA_LITE_DOCKER_CREATE_CEILING", "64"))
    ratio = nvme if host_r.has_nonrotational_storage else hdd
    return max(floor_n, min(int(host_r.vcpu * ratio), ceiling_n))


def _env_override(name: str, derived: int, ceiling: int) -> int:
    """Apply an env-var integer override, clamped to ``[1, ceiling]``.

    Malformed values (non-int, empty, negative) are logged once and
    silently fall back to the host-derived value — wrapper scripts
    that do ``export VAR=""`` are common, and crashing the server
    boot on that is unhelpful.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return derived
    try:
        v = int(raw)
    except ValueError:
        logger.warning(
            "ignoring malformed %s=%r; falling back to derived %d",
            name, raw, derived,
        )
        return derived
    return max(1, min(v, ceiling))


def derive_reset_concurrency() -> int:
    """Cap on concurrent ``env.reset()`` work.

    Reset is usually shorter than full container/VM construction, so the host can
    sustain higher parallelism here. Bounded by
    ``min(2 * docker_create_concurrency, max_live_envs)``.

    Scales tiny→huge:
      * 4-vCPU/16 GB laptop: min(16, 4) = 4
      * 384-vCPU/1505 GB host: min(128, 376) = 128

    Override via ``CUA_LITE_RESET_CONCURRENCY`` env var (still bounded
    by max_live_envs).
    """
    host_r = cached_host_capacity()
    docker_cap = derive_docker_create_concurrency(host=host_r)
    ml = derive_max_live_envs(vcpu=host_r.vcpu, ram_gb=host_r.ram_total_gb)
    derived = min(2 * docker_cap, ml)
    return _env_override("CUA_LITE_RESET_CONCURRENCY", derived, ml)


def derive_admission_config(
    host: HostCapacity,
    **overrides,
) -> AdmissionConfig:
    """Production-default :class:`AdmissionConfig` derived from *host*.

    ``host`` is required — the sole caller (``scripts/serve_env.py``)
    already probes :func:`cached_host_capacity` for its own logging, so
    a second implicit probe here would just be a divergent second
    source of the same facts.

    Test fixtures construct :class:`AdmissionConfig` directly with
    explicit values so unit tests stay deterministic; production paths
    (``scripts/serve_env.py``) call this so any new scaling formula
    lands in one place.

    Env-var escape hatches are applied before explicit ``overrides`` so CLI
    flags remain the final word.
    """
    GB = 1024 ** 3
    defaults: dict = {
        "max_live_envs": derive_max_live_envs(
            vcpu=host.vcpu, ram_gb=host.ram_total_gb
        ),
        "emergency_ram_pct": 95.0,
        # Scale absolute floor with host RAM: 1% with a hard 2 GB minimum.
        # A 2 GB literal on a 1.5 TB host is 0.13% — effectively
        # disabled, so this scaling closes a long-standing hole.
        "emergency_ram_free_min_bytes": max(
            2 * GB, int(host.ram_total_gb * 0.01 * GB)
        ),
        "emergency_load_per_cpu": 2.5,
        # Same scaling philosophy for disk: 1% with a 10 GB floor.
        "emergency_disk_free_min_bytes": max(
            10 * GB, int(host.disk_total_gb * 0.01 * GB)
        ),
        # Any swap use is trouble — kernel is already paging, latency
        # has spiked, and the env-server's stall budget is gone.
        "emergency_swap_pct": 1.0,
        "docker_create_concurrency": derive_docker_create_concurrency(host=host),
    }
    defaults.update(_admission_env_overrides())
    defaults.update(overrides)
    return AdmissionConfig(**defaults)


def log_admission_config(
    config: AdmissionConfig,
    *,
    explicit_max_live_envs: bool,
) -> None:
    """Log the public admission summary; keep threshold details at DEBUG."""
    logger.info(
        "admission: max_live_envs=%d (%s)",
        config.max_live_envs,
        "override" if explicit_max_live_envs else "auto",
    )
    logger.debug(
        "admission details: docker_create_concurrency=%d; "
        "ram>%.0f%%/free<%.1fGB; disk-free<%.1fGB; swap>%.1f%%; load>%.1f/cpu",
        config.docker_create_concurrency,
        config.emergency_ram_pct,
        config.emergency_ram_free_min_bytes / (1024 ** 3),
        config.emergency_disk_free_min_bytes / (1024 ** 3),
        config.emergency_swap_pct,
        config.emergency_load_per_cpu,
    )


def _admission_env_overrides() -> dict:
    overrides: dict = {}
    for env_var, key, conv in (
        ("CUA_LITE_EMERGENCY_RAM_PCT", "emergency_ram_pct", float),
        ("CUA_LITE_EMERGENCY_LOAD_PER_CPU", "emergency_load_per_cpu", float),
        ("CUA_LITE_EMERGENCY_SWAP_PCT", "emergency_swap_pct", float),
        ("CUA_LITE_DOCKER_CREATE_CONCURRENCY", "docker_create_concurrency", int),
    ):
        value = os.environ.get(env_var)
        if value is not None:
            overrides[key] = conv(value)

    GB = 1024 ** 3
    for env_var, key in (
        ("CUA_LITE_EMERGENCY_RAM_FREE_MIN_GB", "emergency_ram_free_min_bytes"),
        ("CUA_LITE_EMERGENCY_DISK_FREE_MIN_GB", "emergency_disk_free_min_bytes"),
    ):
        value = os.environ.get(env_var)
        if value is not None:
            overrides[key] = int(float(value) * GB)
    return overrides


# --- L1 plugin registry (extension point) ---

_emergency_checks: list[Callable[[], None]] = []
_emergency_checks_lock = threading.Lock()


def register_emergency_check(fn: Callable[[], None]) -> None:
    """Append a custom L1 check.

    Plugins must be sub-millisecond, raise ``CapacityExhausted`` on
    failure, and be thread-safe (run on the event loop).
    """
    with _emergency_checks_lock:
        _emergency_checks.append(fn)


def _reset_emergency_checks_for_tests() -> None:
    with _emergency_checks_lock:
        _emergency_checks.clear()


# --- Process-wide docker semaphore (structural, not admission) ---

_docker_sema: threading.Semaphore | None = None
_docker_sema_capacity: int = 0
_docker_sema_init_lock = threading.Lock()
_docker_sema_timeout_total: int = 0

#: Count of in-flight callers currently blocked in ``acquire()`` —
#: used to estimate queue depth for adaptive retry-after on 503.
_docker_sema_pending: int = 0
_docker_sema_pending_lock = threading.Lock()

#: EMA of observed ``docker_create_slot`` body duration (one full create
#: cycle from acquire → release). Seeded at 20 s — a realistic cold-boot
#: time for a docker-backed env like lite.osworld so the first 503's
#: retry hint is sane before any real data has accumulated.
_docker_create_p50_s: float = 20.0
_DOCKER_CREATE_EMA_ALPHA: float = 0.1
_docker_create_p50_lock = threading.Lock()


# --- Host-sensor cache (perf: amortize psutil syscalls across burst) ---

#: TTL for the cached L1 host sensors (RAM/swap/disk/load). 500 ms
#: trades sub-second admission reactivity for amortizing the ~150 µs
#: of psutil syscalls across every request in a burst — at 100 req/s
#: this saves ~15 ms/s of event-loop blocking. Tune via env var if
#: faster L1 reaction is needed.
_HOST_SENSOR_TTL_S: float = float(
    os.environ.get("CUA_LITE_HOST_SENSOR_TTL_S", "0.5")
)


@dataclass(frozen=True)
class _HostSensors:
    """One probe of every L1 sensor — RAM/swap/disk/load — used by
    :meth:`AdmissionGate._check_*` and :meth:`AdmissionGate.snapshot`.
    """
    ram_pct: float
    ram_available_bytes: int
    swap_pct: float
    disk_free_bytes: int
    load_1m: float  # raw — divide by effective_vcpu at use site


_host_sensor_cache: tuple[float, _HostSensors] | None = None
_host_sensor_cache_lock = threading.Lock()


def _read_host_sensors() -> _HostSensors:
    """Cached L1 sensor read. Double-checked under
    :data:`_host_sensor_cache_lock` so concurrent first-callers don't
    each pay the syscall cost.
    """
    global _host_sensor_cache
    cache = _host_sensor_cache
    now = time.monotonic()
    if cache is not None and now - cache[0] < _HOST_SENSOR_TTL_S:
        return cache[1]
    with _host_sensor_cache_lock:
        cache = _host_sensor_cache
        if cache is not None and time.monotonic() - cache[0] < _HOST_SENSOR_TTL_S:
            return cache[1]
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        du = psutil.disk_usage("/")
        load_1m = os.getloadavg()[0]
        sensors = _HostSensors(
            ram_pct=vm.percent,
            ram_available_bytes=vm.available,
            swap_pct=sm.percent,
            disk_free_bytes=du.free,
            load_1m=load_1m,
        )
        _host_sensor_cache = (time.monotonic(), sensors)
        return sensors


def _record_docker_create_duration(duration_s: float) -> None:
    """Update the EMA of observed docker-create duration.

    Cheap, single-mutation update — locked because this is called from
    both the asyncio loop (via ``asyncio.to_thread``) and the sync
    container factories on worker threads.
    """
    global _docker_create_p50_s
    if duration_s <= 0:
        return
    with _docker_create_p50_lock:
        _docker_create_p50_s = (
            _DOCKER_CREATE_EMA_ALPHA * duration_s
            + (1 - _DOCKER_CREATE_EMA_ALPHA) * _docker_create_p50_s
        )


def _docker_sema_retry_after_s() -> int:
    """Adaptive retry-after for a docker-sema 503.

    Estimates the wait until a slot frees as ``observed_p50 ×
    queue_depth / sema_capacity`` so a host under heavy queue gives
    longer hints than one with a brief spike. Adds ±30% jitter to
    spread waiters' wake times across a window (otherwise all 503'd
    clients return at exactly the same instant and re-saturate the
    sema).

    Clamped to ``[EMERGENCY_RETRY_AFTER_S, 120s]`` so a transient
    blip never tells the client to wait minutes, and a deep wedge
    never tells it to come back instantly.
    """
    if _docker_sema is None:
        return EMERGENCY_RETRY_AFTER_S
    capacity = max(1, _docker_sema_capacity)
    # ``_docker_sema_pending`` already excludes "got a slot, doing work" —
    # it's the number of callers stuck in ``acquire()`` right now.
    queue = max(1, _docker_sema_pending)
    with _docker_create_p50_lock:
        p50 = _docker_create_p50_s
    base = p50 * queue / capacity
    jitter = random.uniform(0.7, 1.3)
    return int(max(EMERGENCY_RETRY_AFTER_S, min(base * jitter, 120)))


def configure_docker_sema(concurrency: int) -> None:
    """Install the process-wide docker semaphore once at startup.

    Subsequent calls with the same concurrency are no-ops; different
    concurrency raises (restart to change).
    """
    global _docker_sema, _docker_sema_capacity
    with _docker_sema_init_lock:
        if _docker_sema is None:
            _docker_sema = threading.Semaphore(concurrency)
            _docker_sema_capacity = concurrency
        elif _docker_sema_capacity != concurrency:
            raise RuntimeError(
                f"docker semaphore already configured with capacity "
                f"{_docker_sema_capacity}; cannot re-configure to "
                f"{concurrency}. Restart the process to change."
            )


def _docker_in_flight() -> int:
    if _docker_sema is None:
        return 0
    return _docker_sema_capacity - _docker_sema._value


@contextmanager
def docker_create_slot() -> Generator[None, None, None]:
    """Sync context manager for ``docker run`` serialization.

    THE CONDITION for holding it (not a caller list — a list here goes stale
    the next time an env is added, and then reads as a restriction): hold it
    around any container boot that shells out to ``docker run`` synchronously
    from a worker thread. In practice that is the ``start(c)`` callback of a
    DEDICATED acquire — :func:`lite.gym.container.boot_with_retry` documents
    the same rule from the other side — and the hold spans the WHOLE boot
    (create + readiness wait), not just the ``docker run`` call, because the
    disk-IO burst behind it is the thing being serialized.

    Sync, not async, for the same reason: the callers are already off the
    event loop in an ``asyncio.to_thread`` worker.

    Bounded wait — raises :class:`CapacityExhausted` if the daemon is
    so backed up that :data:`DOCKER_CREATE_WAIT_TIMEOUT_S` elapses
    without a slot. ``retry_after_s`` is computed from the observed
    create p50 × current queue depth so deeper queues self-pace the
    client. No-op if the semaphore was never configured (direct mode —
    see the module docstring).
    """
    global _docker_sema_timeout_total, _docker_sema_pending
    if _docker_sema is None:
        yield
        return
    with _docker_sema_pending_lock:
        _docker_sema_pending += 1
    try:
        acquired = _docker_sema.acquire(timeout=DOCKER_CREATE_WAIT_TIMEOUT_S)
    finally:
        with _docker_sema_pending_lock:
            _docker_sema_pending -= 1
    if not acquired:
        _docker_sema_timeout_total += 1
        raise CapacityExhausted(
            f"docker daemon slot unavailable after "
            f"{DOCKER_CREATE_WAIT_TIMEOUT_S:.0f}s "
            f"(daemon back-pressured or wedged)",
            retry_after_s=_docker_sema_retry_after_s(),
            layer="docker_sema",
        )
    start = time.monotonic()
    try:
        yield
    finally:
        _record_docker_create_duration(time.monotonic() - start)
        _docker_sema.release()


@asynccontextmanager
async def docker_create_slot_async() -> AsyncGenerator[None, None]:
    """Async docker-slot wrapper sharing the threading semaphore.

    Cancellation safety: a CancelledError during acquire waits for
    the worker thread to settle and releases the slot if it landed
    one — keeps the counter honest under request-level cancellation.

    No-op if the semaphore was never configured (direct mode — see the module
    docstring). ``sandbox/base.py`` additionally skips this call entirely when it
    has no ``env_id``, so direct mode does not even reach here.
    """
    global _docker_sema_timeout_total, _docker_sema_pending
    if _docker_sema is None:
        yield
        return
    with _docker_sema_pending_lock:
        _docker_sema_pending += 1
    acquire_task = asyncio.ensure_future(asyncio.to_thread(
        _docker_sema.acquire, True, DOCKER_CREATE_WAIT_TIMEOUT_S,
    ))
    try:
        try:
            acquired = await acquire_task
        except asyncio.CancelledError:
            # The blocking ``threading.Semaphore.acquire`` running on a
            # worker thread doesn't honor asyncio cancellation — it will
            # eventually complete (success or timeout). The previous
            # ``await asyncio.shield(acquire_task)`` here was vulnerable
            # to a SECOND CancelledError during the shielded wait
            # (uvicorn lifespan cascade), in which case the worker
            # thread could still acquire the slot but no one would
            # release it. Instead, install a done-callback that
            # releases the slot iff the late arrival did acquire, and
            # re-raise the cancellation immediately.
            def _release_on_late_acquire(task: asyncio.Task) -> None:
                try:
                    if (not task.cancelled()
                            and task.exception() is None
                            and task.result()):
                        if _docker_sema is not None:
                            _docker_sema.release()
                except Exception:
                    pass
            acquire_task.add_done_callback(_release_on_late_acquire)
            raise
    finally:
        with _docker_sema_pending_lock:
            _docker_sema_pending -= 1
    if not acquired:
        _docker_sema_timeout_total += 1
        raise CapacityExhausted(
            f"docker daemon slot unavailable after "
            f"{DOCKER_CREATE_WAIT_TIMEOUT_S:.0f}s "
            f"(daemon back-pressured or wedged)",
            retry_after_s=_docker_sema_retry_after_s(),
            layer="docker_sema",
        )
    start = time.monotonic()
    try:
        yield
    finally:
        _record_docker_create_duration(time.monotonic() - start)
        _docker_sema.release()


# --- AdmissionGate ---

class AdmissionGate:
    """Three-layer admission for a single env-server instance.

    ``in_flight`` is mutated only from the uvicorn event loop; L1
    checks are sync and never await.
    """

    def __init__(self, config: AdmissionConfig) -> None:
        self.config = config
        self._in_flight: int = 0
        self._503_counters: dict[str, int] = {
            "emergency": 0,
            "capacity": 0,
            "env_internal": 0,
        }
        # Cache cgroup-aware vCPU count once — used by ``_check_load``
        # and ``snapshot``. Probing each call would re-walk cgroup
        # files for no value (host topology doesn't change).
        self._effective_vcpu: int = max(1, effective_cpu_count())
        configure_docker_sema(config.docker_create_concurrency)
        # ``_check_disk`` and ``_check_swap`` no-op when their floors
        # are at the "disabled" sentinel — keeps legacy fixtures green
        # while production picks up the new checks via
        # ``derive_admission_config``.
        self._builtin_checks = (
            self._check_ram,
            self._check_load,
            self._check_disk,
            self._check_swap,
        )

    # --- L1 builtin sensors ---

    def _check_ram(self) -> None:
        s = _read_host_sensors()
        if s.ram_pct > self.config.emergency_ram_pct:
            raise CapacityExhausted(
                f"host RAM {s.ram_pct:.1f}% > {self.config.emergency_ram_pct:.0f}%",
                retry_after_s=EMERGENCY_RETRY_AFTER_S,
                layer="emergency",
            )
        if s.ram_available_bytes < self.config.emergency_ram_free_min_bytes:
            free_gb = s.ram_available_bytes / (1024 ** 3)
            min_gb = self.config.emergency_ram_free_min_bytes / (1024 ** 3)
            raise CapacityExhausted(
                f"host free RAM {free_gb:.1f} GB < {min_gb:.1f} GB floor",
                retry_after_s=EMERGENCY_RETRY_AFTER_S,
                layer="emergency",
            )

    def _check_load(self) -> None:
        # cgroup-aware divisor: cached at gate-init so the check is
        # arithmetic-only (the load itself comes from the host-sensor
        # cache below).
        s = _read_host_sensors()
        load_per_cpu = s.load_1m / self._effective_vcpu
        if load_per_cpu > self.config.emergency_load_per_cpu:
            raise CapacityExhausted(
                f"host load {s.load_1m:.1f} ({load_per_cpu:.2f}/cpu) "
                f"> {self.config.emergency_load_per_cpu:.1f}/cpu",
                retry_after_s=EMERGENCY_RETRY_AFTER_S,
                layer="emergency",
            )

    def _check_disk(self) -> None:
        if self.config.emergency_disk_free_min_bytes <= 0:
            return  # disabled by fixture / opt-out
        s = _read_host_sensors()
        if s.disk_free_bytes < self.config.emergency_disk_free_min_bytes:
            free_gb = s.disk_free_bytes / (1024 ** 3)
            min_gb = self.config.emergency_disk_free_min_bytes / (1024 ** 3)
            raise CapacityExhausted(
                f"host free disk {free_gb:.1f} GB < {min_gb:.1f} GB floor",
                retry_after_s=EMERGENCY_RETRY_AFTER_S,
                layer="emergency",
            )

    def _check_swap(self) -> None:
        if self.config.emergency_swap_pct >= 100.0:
            return  # disabled
        # ``swap_memory().total`` can be 0 on swap-less hosts (most
        # cloud VMs) — ``.percent`` is then defined as 0.0 by psutil,
        # so the check just no-ops on those hosts. Good.
        s = _read_host_sensors()
        if s.swap_pct > self.config.emergency_swap_pct:
            raise CapacityExhausted(
                f"host swap {s.swap_pct:.1f}% > "
                f"{self.config.emergency_swap_pct:.1f}% (kernel is paging)",
                retry_after_s=EMERGENCY_RETRY_AFTER_S,
                layer="emergency",
            )

    # --- L1 + L2 admission ---

    def require_capacity(self) -> None:
        """Run L1 (builtins + plugins) and L2. Raises CapacityExhausted.

        ``require_capacity()`` and ``admit()`` are separate so the server can acquire
        conflict keys and surface the precise capacity layer before consuming
        an L2 slot.
        """
        for fn in self._builtin_checks:
            try:
                fn()
            except CapacityExhausted:
                self._503_counters["emergency"] += 1
                raise
        with _emergency_checks_lock:
            plugins = list(_emergency_checks)
        for fn in plugins:
            try:
                fn()
            except CapacityExhausted as e:
                e.layer = "emergency"
                self._503_counters["emergency"] += 1
                raise
        # L2.
        if self._in_flight >= self.config.max_live_envs:
            self._503_counters["capacity"] += 1
            raise CapacityExhausted(
                f"live env cap reached "
                f"({self._in_flight}/{self.config.max_live_envs})",
                retry_after_s=EMERGENCY_RETRY_AFTER_S,
                layer="capacity",
            )

    def admit(self) -> None:
        """Increment in_flight; pair with one ``release()`` per admit."""
        self._in_flight += 1

    def release(self) -> None:
        """Decrement in_flight."""
        self._in_flight -= 1

    def record_env_internal_503(self) -> None:
        """Bump the L3 counter (called by the FastAPI handler)."""
        self._503_counters["env_internal"] += 1

    # --- Observability ---

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def snapshot(self) -> dict[str, int | float | None]:
        """Read-only state for ``/metrics``."""
        # Same cached sensors as ``_check_*`` — keeps the metric
        # self-consistent with what L1 most recently saw, and means a
        # Prometheus scrape during burst pays zero psutil cost.
        s = _read_host_sensors()
        with _docker_create_p50_lock:
            p50 = _docker_create_p50_s
        return {
            "max_live_envs": self.config.max_live_envs,
            "in_flight": self._in_flight,
            "emergency_503_total": self._503_counters["emergency"],
            "capacity_503_total": self._503_counters["capacity"],
            "docker_sema_503_total": _docker_sema_timeout_total,
            "env_internal_503_total": self._503_counters["env_internal"],
            "docker_in_flight": _docker_in_flight(),
            "docker_sema_pending": _docker_sema_pending,
            "docker_create_p50_s": p50,
            "host_ram_percent": s.ram_pct,
            "host_ram_free_bytes": s.ram_available_bytes,
            "host_swap_percent": s.swap_pct,
            "host_disk_free_bytes": s.disk_free_bytes,
            "host_load_per_cpu": s.load_1m / self._effective_vcpu,
        }
