"""Capacity-aware sizing for the WebGym OmniBoxes pool (runs HOST-side).

webgym is container-only: ``WebGymContainerServices.ensure`` calls
``derive_pool_size_from_env`` HOST-side, BEFORE ``docker run``, to size the
in-container pool ``M``, then passes the result as ``docker run -e
WEBGYM_INSTANCES=<M>`` (read by ``docker/entrypoint.sh``'s ``deploy.py M``).
Because it runs before the container exists, the capacity probe
(``cached_host_capacity``) sees the HOST's RAM/vCPU (not a container cgroup
share) when M is auto-derived.

webgym's bottleneck is node DEGRADATION (a long-lived node serializes/hangs on
concurrent ``/screenshot``), not GPT (2-4s/turn) or raw compute (a fresh node
serves 64-concurrent screenshots at ~1s, 0-fail). So the throughput recipe is: a
FRESH pool, sized to available capacity, with enough stateless node front-ends
that no single front-end saturates.

Two INDEPENDENT sizing values (the node server is a stateless by-port proxy over a SHARED
redis instance pool — see ``omniboxes/node/server.py``: ``port = redis.spop(...)``
→ proxy to ``127.0.0.1:{port}``; instances are NOT partitioned per node):

* ``instances`` (M) — total ``instance_server`` browsers. This IS the concurrency
  ceiling (each runs one trajectory at a time). Auto-sized to available RAM/vCPU.
* ``nodes`` (N) — master→node front-end processes. **Defaults to 1.** The node
  server is a stateless by-port proxy over the SHARED redis pool, so N>1 on ONE
  host (one redis) makes each front-end report the FULL pool → the master
  double-counts capacity and over-routes. OmniBoxes' native multi-node is
  multi-MACHINE only (per-machine redis db0 = that node's partition). So on a
  single host N stays 1; the native single-node ``deploy.py`` already serves M
  fine (one front-end handled 64-concurrent at ~1s). The override exists for a
  real multi-machine deploy.

Configure ``instances`` / ``target_concurrency`` in ``configs/default.yaml`` or a
``WEBGYM_CONFIG`` replacement. ``WEBGYM_INSTANCES`` is only the internal value
``WebGymContainerServices.ensure`` passes into ``docker run`` after resolving the
yaml/default sizing.

Run (unit test):
``uv run pytest tests/gym/envs/webgym/test_pool_sizing.py -k pool_sizing``
"""
from __future__ import annotations

from dataclasses import dataclass

# A Chromium instance_server + its page is ~0.5 GB resident under load.
_INSTANCE_RAM_GB = 0.5
# Spend at most this fraction of host RAM on the browser fleet (leave headroom
# for the agent process, GPT client, co-resident jobs on a shared host).
_RAM_BUDGET_FRAC = 0.30
# Also bound by vCPU: a browser doing layout/screenshot is bursty but cheap when
# idle-waiting on the agent; allow this many instances per vCPU.
_INSTANCES_PER_VCPU = 1.5
# Sanity caps. Upper bound is the largest pool the NATIVE single-node deploy.py
# Sanity cap on auto-sized M. NOTE: bring-up is no longer the limiter — with the
# fast-port-clear patch (docker/patches/process_manager.py) a 256-pool warms in
# ~10s and serves fine (was: 256 timed out in the old O(M) lsof pre-flight). The
# cap is now a politeness + diminishing-returns choice: the webgym workload is
# per-trajectory LATENCY-bound (GPT turns + real-site nav), not CPU-bound, so
# concurrency past what keeps the agent/GPT pipeline busy mostly adds browser
# footprint for little gain. 128 is ≈2× the old 64 default and validated to warm
# AND serve. Operators can raise M via yaml ``server_kwargs.instances``; the
# service forwards the resolved value to the container as ``WEBGYM_INSTANCES``.
_MAX_INSTANCES = 128
_MIN_INSTANCES = 8


@dataclass(frozen=True)
class PoolSize:
    instances: int  # total browsers (M) — the concurrency ceiling
    nodes: int      # stateless front-end processes (N)


# A pool only ever holds ~``concurrency`` leases at once (one ``instance_server``
# per in-flight trajectory). Auto-sizing M from host RAM/vCPU alone over-
# provisions when concurrency is modest — e.g. host → 128 but a collect runs at
# conc 56, so ~2.3× the browsers idle, wasting RAM AND enlarging the crash
# surface for zero throughput gain (webgym is per-trajectory LATENCY-bound, not
# host-CPU-bound). When the intended concurrency is known, cap M just above it.
_CONCURRENCY_HEADROOM = 1.3  # M ≈ ceil(conc × 1.3): leases + reaper/`/get`-race slack


def derive_pool_size(
    vcpu: int,
    ram_total_gb: float,
    *,
    instances_override: int | None = None,
    nodes_override: int | None = None,
    concurrency_hint: int | None = None,
) -> PoolSize:
    """Return the (instances, nodes) pool size for a host. Pure function —
    deterministic in its args so it's unit-testable without a real host.

    ``concurrency_hint`` (when > 0 and no explicit ``instances_override``) caps
    the host-derived M at ``ceil(concurrency_hint × _CONCURRENCY_HEADROOM)`` —
    right-sizing the fleet to the workload instead of over-provisioning to host
    capacity. Unset/0 preserves the legacy host-only sizing exactly."""
    import math
    if instances_override is not None and instances_override > 0:
        instances = instances_override
    else:
        ram_cap = int((ram_total_gb * _RAM_BUDGET_FRAC) / _INSTANCE_RAM_GB)
        cpu_cap = int(vcpu * _INSTANCES_PER_VCPU)
        instances = max(_MIN_INSTANCES, min(ram_cap, cpu_cap, _MAX_INSTANCES))
        if concurrency_hint is not None and concurrency_hint > 0:
            conc_cap = math.ceil(concurrency_hint * _CONCURRENCY_HEADROOM)
            instances = max(_MIN_INSTANCES, min(instances, conc_cap))

    # Default N=1. OmniBoxes' node server is a STATELESS by-port proxy over a
    # SHARED redis instance pool, so multiple node front-ends on ONE host (one
    # redis) each report the FULL pool to the master → capacity double-counts and
    # the master over-routes (verified). Native multi-node is multi-MACHINE only
    # (per-machine redis db0 = that node's partition; master redis db1 = registry)
    # — not applicable to our single big host. And N is not the throughput lever:
    # M (instances) is; one front-end serves 64-concurrent at ~1s. The override is
    # kept for a future per-node-redis-partition deploy or a real multi-machine run.
    nodes = nodes_override if (nodes_override and nodes_override > 0) else 1
    return PoolSize(instances=instances, nodes=nodes)


def derive_pool_size_from_env() -> PoolSize:
    """Production entry point: pool size M from the yaml, then probe host capacity.

    M = the yaml ``server_kwargs.instances`` (``0`` = auto-derive from host). Change
    it by editing ``configs/default.yaml`` or ``WEBGYM_CONFIG=<other>.yaml`` — there is
    no per-knob env override (the universal env-config protocol; ``WEBGYM_INSTANCES``
    survives only as the internal value ``ensure`` hands to ``docker run -e`` →
    the container's ``deploy.py M``).
    Falls back to a safe single-node default if the capacity probe is unavailable."""
    from pathlib import Path

    from lite.gym.utils import config as env_config

    _sk = env_config.load(
        str(Path(__file__).resolve().parents[0])
    ).server_kwargs
    _yaml_inst = _sk.get("instances", 0)
    inst_override = _yaml_inst if _yaml_inst else None
    # Optional right-sizing knob: cap auto-sized M just above intended
    # concurrency (0/absent = legacy host-only sizing). Ignored when
    # ``instances`` is explicitly pinned.
    _conc_hint = _sk.get("target_concurrency", 0) or None
    nodes_override = None  # single front-end (N=1); see module docstring
    try:
        from lite.gym.utils.server.capacity import cached_host_capacity
        host = cached_host_capacity()
        vcpu, ram = host.vcpu, host.ram_total_gb
    except Exception:
        vcpu, ram = 8, 32.0  # conservative fallback
    return derive_pool_size(
        vcpu, ram,
        instances_override=inst_override,
        nodes_override=nodes_override,
        concurrency_hint=_conc_hint,
    )
