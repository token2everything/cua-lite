"""Single source of truth for host capacity.

Probed once at process startup, cached, then queried by:

* :class:`~lite.gym.remote.admission.AdmissionConfig` —
  L1 emergency floors (``_check_disk``, ``_check_swap``), L2 max-live cap,
  structural docker semaphore size.
* env-internal L3 pools (e.g. mobilegym's Chromium browser count).
* :mod:`lite.gym.remote.client` — 503 retry deadline calibration.

Replaces hardcoded defaults that were previously scattered across
``admission.py`` (8 docker slots, 2 GB RAM floor), ``mobilegym/main.py``
(16 browsers), and ``scripts/serve_env.py``.

Probe-only: this module holds static host facts. Dynamic state (current
free RAM, current disk free, load) is queried at runtime by the L1
checks themselves — caching that would defeat its purpose.

Usage::

    from lite.gym.utils.server.capacity import cached_host_capacity
    host = cached_host_capacity()           # cached probe
    sema = max(8, min(host.vcpu // 8, 64))
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)

_GB = 1024 ** 3


def effective_cpu_count() -> int:
    """Cgroup-aware CPU count for this process.

    Falls back through Python 3.13+ ``os.process_cpu_count()`` → cgroup
    v2 → cgroup v1 → host ``os.cpu_count()``. A container limited to 4
    CPUs on a 96-CPU host must size for 4.
    """
    # 1. Python 3.13+ native cgroup-aware (PEP 745).
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        try:
            n = process_cpu_count()
            if n is not None and n > 0:
                return int(n)
        except OSError:
            pass

    # 2. cgroup v2: ``max <period>`` or ``<quota> <period>``
    try:
        with open("/sys/fs/cgroup/cpu.max", "r") as f:
            content = f.read().strip()
        if content and not content.startswith("max"):
            quota_s, period_s = content.split()
            quota = int(quota_s)
            period = int(period_s)
            if quota > 0 and period > 0:
                # Round up so a 0.5-CPU limit yields 1 (avoids 0).
                return max(1, -(-quota // period))
    except (OSError, ValueError):
        pass

    # 3. cgroup v1: cfs_quota_us / cfs_period_us
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f:
            quota = int(f.read().strip())
        if quota > 0:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as f:
                period = int(f.read().strip())
            if period > 0:
                return max(1, -(-quota // period))
    except (OSError, ValueError):
        pass

    # 4. Host CPU count fallback.
    return os.cpu_count() or 1


def _detect_nonrotational_storage() -> bool:
    """Return True if at least one non-rotational (SSD/NVMe) block device exists.

    Used to bias docker-create-concurrency: image extraction is heavily
    disk-I/O bound, and NVMe sustains ~4-10× the parallel throughput of
    spinning rust.

    Heuristic, not strict: we don't try to identify the *specific* device
    backing ``/var/lib/docker`` (rootless docker complicates that). The
    assumption is that ops who deliberately put docker data on slow
    media on a host with mixed storage will override
    ``CUA_LITE_DOCKER_CREATE_CONCURRENCY`` explicitly. False is the
    conservative side ("treat as HDD").
    """
    try:
        for p in Path("/sys/block").glob("*/queue/rotational"):
            dev = p.parts[-3]
            # Skip pseudo-devices that don't represent real persistent storage.
            if dev.startswith(("loop", "dm-", "ram", "sr", "fd")):
                continue
            try:
                if p.read_text().strip() == "0":
                    return True
            except OSError:
                continue
        return False
    except Exception:
        return False


@dataclass(frozen=True)
class HostCapacity:
    """Immutable snapshot of static host facts. Probe once at startup."""

    #: Effective vCPU available to *this* process (cgroup-aware).
    vcpu: int
    #: Total host RAM (GB). Used to scale RAM-relative floors.
    ram_total_gb: float
    #: Total disk on the root filesystem (GB). Used to scale disk floor.
    disk_total_gb: float
    #: Heuristic: True if the host has at least one non-rotational
    #: block device. Used to bias docker-create-concurrency.
    has_nonrotational_storage: bool

    @classmethod
    def probe(cls) -> HostCapacity:
        vm = psutil.virtual_memory()
        # ``/`` is a reasonable proxy for "where docker writes": images
        # and container state both land there by default. Operators who
        # bind-mount /var/lib/docker (or its rootless equivalent) onto a
        # separate volume should override the disk floor explicitly.
        du = psutil.disk_usage("/")
        return cls(
            vcpu=effective_cpu_count(),
            ram_total_gb=vm.total / _GB,
            disk_total_gb=du.total / _GB,
            has_nonrotational_storage=_detect_nonrotational_storage(),
        )


@lru_cache(maxsize=1)
def cached_host_capacity() -> HostCapacity:
    """Module-level cached probe. Computed exactly once per process.

    Callers don't need to thread a ``HostCapacity`` instance through ---
    just call this. Tests that need a controlled probe should construct
    ``HostCapacity(...)`` directly and pass it to the function that
    accepts it as an argument; never patch this cache.
    """
    host = HostCapacity.probe()
    logger.info(
        "host capacity: vcpu=%d ram=%.0fGB disk=%.0fGB nonrotational=%s",
        host.vcpu, host.ram_total_gb, host.disk_total_gb,
        host.has_nonrotational_storage,
    )
    return host
