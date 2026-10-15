"""Fleet router: one client-facing endpoint that fans a fleet of env-server
nodes (:mod:`lite.gym.remote.server`) behind the SAME wire contract, so
:class:`lite.gym.remote.client.LiteEnvClient` points at the router with zero
client changes.

Library half of the fleet split (mirrors the server/serve_env split):
  - This module: Node / FleetState / providers / ``make_fleet_app``.
  - Thin launcher: [`scripts/serve_fleet.py`](/scripts/serve_fleet.py).

Design (static provider):
  * **Sticky routing** — ``POST /instances`` picks the least-outstanding
    eligible node; every ``/instances/{id}/*`` call after that is forwarded to
    the owning node via ``instance_map``. The router NEVER rewrites instance
    ids or mutates payloads. An unknown id (router restart) triggers a fan-out
    ``GET /instances/{id}`` to all healthy nodes WITH THE CALLER'S bearer;
    first 200 repopulates the map; all-miss forwards a typed 404
    ``InstanceGone`` envelope so the client's re-make path fires.
  * **Health** — a background poller probes ``GET /host_status`` per node
    every ``--poll-interval`` and refreshes the env catalog
    (``GET /envs?expand=metadata``) every ``--catalog-interval``, both with
    the router's own ``--node-token`` bearer. Lazy corrections dominate:
    a create-503 cools the node down per its Retry-After; a connect failure
    on any forward marks the node unhealthy immediately. Routing never
    blocks on a poll.
  * **Auth** — the caller's ``Authorization`` header is forwarded VERBATIM on
    every proxied call (the nodes do the token scoping). The router itself
    does not gate clients in the MVP; ``--node-token`` is only the poller's
    identity toward the nodes.

Run (see scripts/serve_fleet.py for the CLI)::

    uv run python scripts/serve_fleet.py --port 30300 --nodes-file nodes.txt

with ``nodes.txt`` holding one node per line (``host:port`` or a full URL;
``#`` comments allowed). The file is re-read on mtime change each poll, so
adding a line hot-joins a node.

Unit tests: tests/gym/remote/test_fleet_router.py (in-process fake nodes via
``httpx.ASGITransport`` — no docker, no sockets).
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

import httpx
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from lite.gym.errors import CapacityExhausted, InstanceGone, LiteGymError
from lite.gym.utils.backend.rpc import may_reissue
from lite.utils.registry import split_key

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeouts. The default node-side client timeout must EXCEED the env client's
# 600 s (a reset legitimately takes minutes; the router must not give up
# before the caller does). Health/liveness probes and fan-out lookups use the
# short per-request override.
# ---------------------------------------------------------------------------
NODE_CONNECT_TIMEOUT_S = 10.0
NODE_READ_TIMEOUT_S = 900.0
PROBE_TIMEOUT_S = 10.0
CATALOG_TIMEOUT_S = 120.0  # /envs?expand=metadata can block on task loading

_UNAUTHORIZED_ERR = "401 unauthorized (check --node-token)"

#: Hop-by-hop headers (RFC 7230 §6.1) that a proxy must not forward, plus
#: content-length — the router re-frames every body (Response recomputes it,
#: StreamingResponse chunks), so an upstream length would be a lie.
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length",
    # Not RFC hop-by-hop, but per-hop in practice: uvicorn stamps its own
    # Date/Server on every response, so passing the node's through produced
    # DUPLICATE date/server headers on proxied responses (live-drill finding).
    "date", "server",
})


# =============================================================================
# Providers
# =============================================================================

# KEEP the Protocol even though ``StaticProvider`` is its only public provider:
# it keeps roster loading separate from routing, and ``make_provider`` still
# rejects reserved/unknown provider names loudly for direct-call tests.
class NodeProvider(Protocol):
    """Source of the node roster. ``current_nodes`` is called once per poll
    tick and must be cheap + non-blocking (file stat, cached list, ...)."""

    def current_nodes(self) -> list[tuple[str, str]]:
        """Return ``[(name, base_url), ...]`` for the current roster."""
        ...


class StaticProvider:
    """Roster from a text file: one node per line (``host:port`` or a full
    ``http(s)://`` URL), ``#`` comments and blank lines ignored. Re-parsed
    whenever the file's mtime changes — each poll tick sees edits, so an
    operator hot-joins a node by appending a line."""

    def __init__(self, nodes_file: str | Path) -> None:
        self.nodes_file = Path(nodes_file)
        self._mtime: float | None = None
        self._nodes: list[tuple[str, str]] = []

    def current_nodes(self) -> list[tuple[str, str]]:
        mtime = self.nodes_file.stat().st_mtime
        if mtime != self._mtime:
            self._nodes = self._parse()
            self._mtime = mtime
            logger.info(
                "fleet: nodes-file %s (re)loaded — %d node(s): %s",
                self.nodes_file, len(self._nodes),
                ", ".join(name for name, _ in self._nodes),
            )
        return list(self._nodes)

    def _parse(self) -> list[tuple[str, str]]:
        nodes: list[tuple[str, str]] = []
        for raw in self.nodes_file.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            base_url = line if "://" in line else f"http://{line}"
            base_url = base_url.rstrip("/")
            name = base_url.split("://", 1)[1]  # host[:port] as the node name
            nodes.append((name, base_url))
        return nodes


def make_provider(kind: str, nodes_file: str | None) -> NodeProvider:
    """Provider factory for the launcher and direct-call tests."""
    # Keep loud exits for direct callers: silently defaulting a mistyped provider
    # to static is the failure this branch exists to prevent.
    if kind == "static":
        if nodes_file is None:
            raise SystemExit("--provider static requires --nodes-file")
        return StaticProvider(nodes_file)
    if kind == "slurm":
        raise SystemExit(
            "--provider slurm is not implemented. "
            "Use --provider static with --nodes-file."
        )
    raise SystemExit(f"unknown provider {kind!r}; supported: static, slurm")


# =============================================================================
# State
# =============================================================================

@dataclasses.dataclass
class Node:
    """Router-side view of one env-server node. Mutated by the poller and by
    lazy corrections on the request path; roster membership + ``outstanding``
    are guarded by :attr:`FleetState.lock`, the health/cooldown scalars are
    GIL-atomic writes (race-tolerant, monitoring-grade — same convention as
    server.py's counters)."""

    name: str
    base_url: str
    healthy: bool = False           # flips true on first successful probe
    last_seen: float = 0.0          # monotonic time of last healthy probe
    last_error: str | None = None
    cooldown_until: float = 0.0     # monotonic; set from a create-503's Retry-After
    envs: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    envs_fetched_at: float = 0.0    # 0.0 = catalog never fetched → optimistic
    outstanding: int = 0            # instances the router tracks on this node
    build_info: dict[str, Any] | None = None

    def env_eligible(self, env_id: str) -> bool:
        """Empty catalog cache → eligible (optimistic, catalog is advisory);
        else require the node to report the env available."""
        if not self.envs:
            return True
        meta = self.envs.get(env_id)
        return bool(meta and meta.get("available"))


class FleetState:
    """Roster + sticky instance map + shared HTTP client + counters.

    ``client`` is injectable so unit tests mount in-process fake node apps
    via ``httpx.MockTransport``/``ASGITransport`` mounts; production uses the
    default client with the long node-side timeouts.
    """

    def __init__(
        self,
        provider: NodeProvider,
        *,
        node_token: str = "fleet-router",
        poll_interval_s: float = 10.0,
        catalog_interval_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.provider = provider
        self.node_token = node_token
        self.poll_interval_s = poll_interval_s
        self.catalog_interval_s = catalog_interval_s
        self.client = client if client is not None else httpx.AsyncClient(
            timeout=httpx.Timeout(NODE_READ_TIMEOUT_S, connect=NODE_CONNECT_TIMEOUT_S),
        )
        #: name → Node. Node objects are stable across polls so cooldown /
        #: outstanding survive a roster re-read.
        self.nodes: dict[str, Node] = {}
        #: instance id → owning Node (sticky routing). Populated on 201
        #: create and by fan-out recovery; dropped on DELETE 2xx and on
        #: presumed-lost synthesis.
        self.instance_map: dict[str, Node] = {}
        #: Guards nodes-roster membership + instance_map + outstanding.
        self.lock = asyncio.Lock()
        #: env_id → (fetched_at_monotonic, raw /tasks body). TTL'd response
        #: cache for GET /envs/{id}/tasks (task lists are large + static).
        self.tasks_cache: dict[str, tuple[float, bytes]] = {}
        #: Router-local /metrics counters. Plain Counter — increments are
        #: race-tolerant, monitoring-grade.
        self.counters: Counter[str] = Counter()
        self.started_at = time.time()

    # -- node-facing helpers --------------------------------------------------

    @property
    def _node_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.node_token}"}

    def _map_put(self, id_: str, node: Node) -> None:
        """Caller holds :attr:`lock`. Keeps ``outstanding`` == number of map
        entries pointing at the node."""
        if id_ not in self.instance_map:
            self.instance_map[id_] = node
            node.outstanding += 1

    def _map_pop(self, id_: str) -> None:
        """Caller holds :attr:`lock`."""
        node = self.instance_map.pop(id_, None)
        if node is not None:
            node.outstanding -= 1

    async def healthy_nodes(self) -> list[Node]:
        async with self.lock:
            return [n for n in self.nodes.values() if n.healthy]

    # -- polling ---------------------------------------------------------------

    async def poll_loop(self) -> None:
        while True:
            try:
                await self.poll_once()
            except Exception:
                logger.exception("fleet: poll failed; retrying next tick")
            await asyncio.sleep(self.poll_interval_s)

    async def poll_once(self) -> None:
        """One poll tick: re-read the provider roster (join/leave), probe
        health per node, refresh catalogs on their own (slower) cadence.
        Tests call this directly instead of waiting on the background task."""
        entries = self.provider.current_nodes()
        joined: list[Node] = []
        async with self.lock:
            want = {name for name, _ in entries}
            for name, base_url in entries:
                if name not in self.nodes:
                    node = Node(name=name, base_url=base_url)
                    self.nodes[name] = node
                    joined.append(node)
            for name in set(self.nodes) - want:
                gone = self.nodes.pop(name)
                # Purge sticky entries: the node is out of the roster, so a
                # later touch fan-outs (and 404s typed if truly lost).
                for id_ in [i for i, n in self.instance_map.items() if n is gone]:
                    self._map_pop(id_)
                logger.info("fleet: node %s left the roster", name)
        async with self.lock:
            nodes = list(self.nodes.values())
        now = time.monotonic()
        for node in nodes:
            await self._probe(node)
            catalog_due = (
                node.envs_fetched_at == 0.0
                or now - node.envs_fetched_at >= self.catalog_interval_s
            )
            if node.healthy and catalog_due:
                await self._refresh_catalog(node)

    async def _probe(self, node: Node) -> None:
        try:
            r = await self.client.get(
                f"{node.base_url}/host_status",
                headers=self._node_headers, timeout=PROBE_TIMEOUT_S,
            )
        except httpx.TransportError as e:
            node.healthy = False
            node.last_error = f"host_status: {e}"
            return
        if r.status_code == 200:
            node.healthy = True
            node.last_seen = time.monotonic()
            node.last_error = None
            node.build_info = r.json().get("cua_lite")
        elif r.status_code == 401:
            # Auth mismatch FAILS LOUD (once per flip): silently leaving the
            # node unroutable would read as a health flake, not a config bug.
            if node.last_error != _UNAUTHORIZED_ERR:
                logger.error(
                    "fleet: node %s rejected the router's --node-token (401) "
                    "— node excluded from routing until tokens match",
                    node.name,
                )
            node.healthy = False
            node.last_error = _UNAUTHORIZED_ERR
        else:
            node.healthy = False
            node.last_error = f"host_status HTTP {r.status_code}"

    async def _refresh_catalog(self, node: Node) -> None:
        try:
            r = await self.client.get(
                f"{node.base_url}/envs", params={"expand": "metadata"},
                headers=self._node_headers, timeout=CATALOG_TIMEOUT_S,
            )
        except httpx.TransportError as e:
            logger.debug("fleet: catalog refresh failed for %s: %s", node.name, e)
            return
        if r.status_code == 200:
            node.envs = r.json()
            node.envs_fetched_at = time.monotonic()

    # -- lazy corrections -------------------------------------------------------

    def mark_unreachable(self, node: Node, what: str) -> None:
        node.healthy = False
        node.last_error = what
        self.counters["unreachable_total"] += 1

    async def probe_alive(self, node: Node) -> bool:
        """Liveness re-check used before synthesizing InstanceGone — a single
        failed forward is not proof the node (and its instances) are lost."""
        try:
            r = await self.client.get(
                f"{node.base_url}/host_status",
                headers=self._node_headers, timeout=PROBE_TIMEOUT_S,
            )
            return r.status_code == 200
        except httpx.TransportError:
            return False

    async def recover_instance(self, id_: str, authorization: str | None) -> Node | None:
        """Fan out ``GET /instances/{id}`` to all healthy nodes with the
        CALLER'S bearer (ownership is token-scoped node-side); first 200
        repopulates the sticky map. Returns None when no node knows the id."""
        nodes = await self.healthy_nodes()
        headers = {"Authorization": authorization} if authorization else {}

        async def probe(n: Node) -> bool:
            try:
                r = await self.client.get(
                    f"{n.base_url}/instances/{id_}",
                    headers=headers, timeout=PROBE_TIMEOUT_S,
                )
                return r.status_code == 200
            except httpx.TransportError:
                return False

        hits = await asyncio.gather(*(probe(n) for n in nodes))
        for node, hit in zip(nodes, hits):
            if hit:
                async with self.lock:
                    self._map_put(id_, node)
                self.counters["fanout_recovery_total"] += 1
                logger.info("fleet: recovered instance %s → node %s", id_, node.name)
                return node
        return None

    async def aclose(self) -> None:
        await self.client.aclose()


# =============================================================================
# Wire helpers
# =============================================================================

def _forward_request_headers(request: Request) -> dict[str, str]:
    """Headers the router forwards node-ward: the caller's Authorization
    VERBATIM plus content-type. Everything else (host, content-length,
    accept-encoding negotiation) is owned per-hop by httpx."""
    headers: dict[str, str] = {}
    auth = request.headers.get("authorization")
    if auth:
        headers["Authorization"] = auth
    ct = request.headers.get("content-type")
    if ct:
        headers["Content-Type"] = ct
    return headers


def _forward_response_headers(resp: httpx.Response) -> dict[str, str]:
    return {
        k: v for k, v in resp.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() != "content-type"
    }


def _verbatim(resp: httpx.Response) -> Response:
    """Fully-read node response → client response, byte-identical body +
    status, hop-by-hop headers stripped (framing recomputed)."""
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=_forward_response_headers(resp),
        media_type=resp.headers.get("content-type"),
    )


def _streamed(resp: httpx.Response) -> StreamingResponse:
    """Node response opened with ``stream=True`` → chunked client response.
    ``aiter_raw`` forwards the raw bytes (no transparent decompression, so a
    content-encoding header stays truthful); the node connection is released
    when the client finishes reading."""
    return StreamingResponse(
        resp.aiter_raw(),
        status_code=resp.status_code,
        headers=_forward_response_headers(resp),
        media_type=resp.headers.get("content-type"),
        background=BackgroundTask(resp.aclose),
    )


def _retry_after_s(resp: httpx.Response) -> float:
    try:
        return max(0.0, float(int(resp.headers.get("Retry-After", ""))))
    except ValueError:
        return 5.0


def _typed_envelope(
    exc: LiteGymError, headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Serialize a router-synthesized failure exactly as ``server.py``'s handlers
    do, so its envelope is byte-compatible with a node's and the client
    reconstructs it through ``lite_error_from_payload``."""
    return JSONResponse(
        status_code=exc.http_status,
        content={**exc.to_payload(), "detail": str(exc)},
        headers=headers,
    )


def _instance_gone(what: str) -> JSONResponse:
    """Typed 404 the client reconstructs as :class:`InstanceGone`, firing its
    re-make path."""
    return _typed_envelope(InstanceGone(what))


def _capacity_503(what: str, retry_after_s: float) -> JSONResponse:
    """Typed 503 + ``Retry-After`` for a router-level "no node can take this
    right now" — the same transient the nodes raise, so the client's existing
    capacity wait/retry loop applies without a router-specific branch."""
    exc = CapacityExhausted(what, retry_after_s=retry_after_s, layer="fleet")
    return _typed_envelope(
        exc, headers={"Retry-After": str(max(1, int(round(exc.retry_after_s))))},
    )


# =============================================================================
# App
# =============================================================================

def make_fleet_app(fleet: FleetState) -> FastAPI:
    """Build the router FastAPI app bound to a :class:`FleetState`.

    The background poller runs via the app lifespan; unit tests drive
    ``fleet.poll_once()`` directly (``httpx.ASGITransport`` doesn't run
    lifespan), so routing logic is testable without timers."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        poller = asyncio.create_task(fleet.poll_loop(), name="fleet-poller")
        try:
            yield
        finally:
            poller.cancel()
            try:
                await poller
            except (asyncio.CancelledError, Exception):
                pass
            await fleet.aclose()

    app = FastAPI(title="cua-lite fleet router", lifespan=lifespan)

    # -- create (routed) ------------------------------------------------------

    @app.post("/instances")
    async def create_instance(request: Request) -> Response:
        body = await request.body()
        try:
            env_id, _ = split_key(json.loads(body)["env_key"])
        except Exception:
            raise HTTPException(
                status_code=400, detail="body must be JSON with an env_key field",
            )
        fleet.counters["create_total"] += 1
        now = time.monotonic()
        async with fleet.lock:
            candidates = sorted(
                (
                    n for n in fleet.nodes.values()
                    if n.healthy and now >= n.cooldown_until and n.env_eligible(env_id)
                ),
                key=lambda n: (n.outstanding, n.name),
            )
        headers = _forward_request_headers(request)
        best_503: tuple[float, httpx.Response] | None = None
        for node in candidates:
            try:
                resp = await fleet.client.post(
                    f"{node.base_url}/instances", content=body, headers=headers,
                )
            except httpx.TransportError as e:
                fleet.mark_unreachable(node, f"create: {e}")
                continue  # cascade — another node can serve this create
            if resp.status_code == 201:
                inst_id = resp.json()["id"]
                async with fleet.lock:
                    fleet._map_put(inst_id, node)
                logger.info(
                    "fleet: created %s on node %s (outstanding=%d)",
                    inst_id, node.name, node.outstanding,
                )
                return _verbatim(resp)
            if resp.status_code == 503:
                fleet.counters["create_503_total"] += 1
                ra = _retry_after_s(resp)
                node.cooldown_until = time.monotonic() + ra
                if best_503 is None or ra < best_503[0]:
                    best_503 = (ra, resp)
                continue  # cascade to the next-least-loaded node
            # Terminal / typed (501 blocked, 400 MakeFailed, ...): the fleet
            # serves ONE homogeneous contract — a typed rejection from one
            # node would repeat on every node, so return it verbatim.
            return _verbatim(resp)
        if best_503 is not None:
            # Every eligible node is saturated: surface the node body VERBATIM
            # (its Retry-After header included) for the node with the MINIMUM
            # hint, so the client backs off the shortest honest amount.
            return _verbatim(best_503[1])
        return _capacity_503(
            f"no eligible fleet node for env_id={env_id!r}",
            fleet.poll_interval_s,
        )

    # -- sticky instance calls --------------------------------------------------

    async def _sticky_forward(request: Request, id_: str, method: str, path: str) -> Response:
        """Forward one instance-scoped call to the node holding it.

        The retry question is :func:`~lite.gym.utils.backend.rpc.may_reissue`'s,
        the same one the client hop and the env↔container hop ask.

        ``replay_safe=False`` on every path, including ``/reset``, and that is a
        policy rather than missing knowledge — each route here does know its own
        idempotence. Two reasons not to thread it: the client hop already owns
        the at-least-once budget for replay-safe ops (4 attempts, jittered
        backoff) and reaches the same node through the same sticky entry, so a
        router resend only multiplies (4 × 2 = 8 requests for one ``/reset``);
        and ``reset_sema`` is per-``env_id`` with capacity > 1, so resending a
        ``/reset`` whose RESPONSE was merely lost can run a second
        ``env.reset()`` concurrently on the same instance.

        So the router resends only what the raiser PROVED never landed.
        """
        async with fleet.lock:
            node = fleet.instance_map.get(id_)
        if node is None:
            node = await fleet.recover_instance(
                id_, request.headers.get("authorization"),
            )
            if node is None:
                fleet.counters["instance_gone_total"] += 1
                return _instance_gone(f"unknown instance id={id_!r} on any fleet node")
        body = await request.body()
        req = fleet.client.build_request(
            method, f"{node.base_url}{path}",
            headers=_forward_request_headers(request), content=body,
        )
        resp: httpx.Response | None = None
        last_exc: Exception | None = None
        for _attempt in range(2):
            try:
                resp = await fleet.client.send(req, stream=True)
                break
            except httpx.TransportError as e:
                last_exc = e
                if not may_reissue(e, replay_safe=False):
                    break
        if resp is None:
            # Forward failed (twice, or once with no licence to re-issue — see
            # the retry note above). Only presume the instance lost when the
            # node ALSO fails a liveness probe — a per-request blip must not
            # 404 a healthy episode.
            if await fleet.probe_alive(node):
                return JSONResponse(
                    status_code=502,
                    content={
                        "error_type": type(last_exc).__name__,
                        "what": f"node {node.name}: {last_exc}",
                    },
                )
            fleet.mark_unreachable(node, f"{method} {path}: {last_exc}")
            async with fleet.lock:
                fleet._map_pop(id_)
            fleet.counters["instance_gone_total"] += 1
            return _instance_gone(
                f"node {node.name} unreachable — instance presumed lost",
            )
        if (method == "DELETE" and resp.status_code < 300) or resp.status_code == 404:
            # DELETE success — or the node itself no longer knows the id
            # (idle-TTL reap / node restart): either way the sticky entry is
            # dead weight and would skew least-outstanding routing forever.
            async with fleet.lock:
                fleet._map_pop(id_)
        return _streamed(resp)

    @app.get("/instances/{id_}")
    async def get_instance(id_: str, request: Request) -> Response:
        return await _sticky_forward(request, id_, "GET", f"/instances/{id_}")

    @app.post("/instances/{id_}/reset")
    async def reset(id_: str, request: Request) -> Response:
        return await _sticky_forward(request, id_, "POST", f"/instances/{id_}/reset")

    @app.post("/instances/{id_}/step")
    async def step(id_: str, request: Request) -> Response:
        return await _sticky_forward(request, id_, "POST", f"/instances/{id_}/step")

    @app.delete("/instances/{id_}")
    async def close(id_: str, request: Request) -> Response:
        return await _sticky_forward(request, id_, "DELETE", f"/instances/{id_}")

    # -- fan-out (list / bulk) ----------------------------------------------------

    async def _fan_out(request: Request, method: str, path: str) -> list[httpx.Response]:
        """Issue the same call to every healthy node with the caller's bearer
        + query string; transport failures mark the node unhealthy and are
        skipped (the merge is over whoever answered)."""
        nodes = await fleet.healthy_nodes()
        headers = _forward_request_headers(request)

        async def one(n: Node) -> httpx.Response | None:
            try:
                return await fleet.client.request(
                    method, f"{n.base_url}{path}",
                    headers=headers, params=request.query_params,
                )
            except httpx.TransportError as e:
                fleet.mark_unreachable(n, f"{method} {path}: {e}")
                return None

        results = await asyncio.gather(*(one(n) for n in nodes))
        return [r for r in results if r is not None]

    def _merge_bodies(responses: list[httpx.Response]) -> Response:
        """Merge 2xx JSON-object bodies: list values concatenate,
        everything else last-writer-wins (the only fanned-out endpoints
        return lists + one bool: /instances' ``instances``; bulk-DELETE's
        ``closed`` / ``skipped_in_flight`` / ``would_close`` + ``dry_run``).
        Non-2xx across the board → first response verbatim (the guard-400
        case is deterministic per query, so every node agrees)."""
        oks = [r for r in responses if r.status_code < 300]
        if not oks:
            if responses:
                return _verbatim(responses[0])
            return _capacity_503("no healthy fleet node", 10.0)
        merged: dict[str, Any] = {}
        for r in oks:
            for k, v in r.json().items():
                if isinstance(v, list):
                    merged.setdefault(k, []).extend(v)
                else:
                    merged[k] = v
        return JSONResponse(content=merged)

    @app.get("/instances")
    async def list_instances(request: Request) -> Response:
        return _merge_bodies(await _fan_out(request, "GET", "/instances"))

    @app.delete("/instances")
    async def bulk_close(request: Request) -> Response:
        resp = _merge_bodies(await _fan_out(request, "DELETE", "/instances"))
        # Purge the closed ids from the sticky map (a router-mediated bulk
        # close otherwise leaves every entry behind, inflating each node's
        # ``outstanding`` for the router's lifetime). ``closed`` is absent on
        # dry_run — nothing to purge then.
        try:
            closed = json.loads(bytes(resp.body)).get("closed") or []
        except Exception:
            closed = []
        if closed:
            async with fleet.lock:
                for id_ in closed:
                    fleet._map_pop(id_)
        return resp

    # -- env catalog (merged from the poller's caches) ------------------------------

    def _merge_env(entries: list[tuple[Node, dict[str, Any]]]) -> dict[str, Any]:
        """available = any node; n_tasks/splits from the FIRST available node
        (the fleet is homogeneous — first is as good as any); error only when
        NO node serves it, concatenating per-node errors prefixed by node name
        so the operator sees where to look."""
        for _node, meta in entries:
            if meta.get("available"):
                return dict(meta)
        errors = "; ".join(
            f"{node.name}: {meta.get('error', 'unavailable')}" for node, meta in entries
        )
        return {"available": False, "error": errors}

    async def _catalog_entries(env_id: str) -> list[tuple[Node, dict[str, Any]]]:
        return [
            (n, n.envs[env_id]) for n in await fleet.healthy_nodes() if env_id in n.envs
        ]

    @app.get("/envs")
    async def list_envs(expand: str | None = None) -> Any:
        nodes = await fleet.healthy_nodes()
        if expand == "metadata":
            merged: dict[str, Any] = {}
            for env_id in sorted({eid for n in nodes for eid in n.envs}):
                merged[env_id] = _merge_env(
                    [(n, n.envs[env_id]) for n in nodes if env_id in n.envs]
                )
            return merged
        if expand is not None:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported expand={expand!r}; supported: 'metadata' or unset",
            )
        return sorted({eid for n in nodes for eid in n.envs})

    @app.get("/envs/{env_id}")
    async def get_env(env_id: str) -> dict[str, Any]:
        entries = await _catalog_entries(env_id)
        if not entries:
            raise HTTPException(
                status_code=404,
                detail=f"env_id={env_id!r} not registered on any fleet node",
            )
        return _merge_env(entries)

    @app.get("/envs/{env_id}/tasks")
    async def env_tasks(env_id: str, request: Request) -> Response:
        now = time.monotonic()
        cached = fleet.tasks_cache.get(env_id)
        if cached is not None and now - cached[0] < fleet.catalog_interval_s:
            return Response(content=cached[1], media_type="application/json")
        # First healthy node with the env available; nodes without a catalog
        # cache yet rank after (optimistic fallback).
        nodes = await fleet.healthy_nodes()
        candidates = [n for n in nodes if n.envs.get(env_id, {}).get("available")]
        candidates += [n for n in nodes if not n.envs]
        if not candidates:
            raise HTTPException(
                status_code=404,
                detail=f"env_id={env_id!r} not available on any fleet node",
            )
        node = candidates[0]
        try:
            resp = await fleet.client.get(
                f"{node.base_url}/envs/{env_id}/tasks",
                headers=_forward_request_headers(request), timeout=CATALOG_TIMEOUT_S,
            )
        except httpx.TransportError as e:
            fleet.mark_unreachable(node, f"/tasks: {e}")
            return _capacity_503(f"node {node.name} unreachable", 10.0)
        if resp.status_code == 200:
            fleet.tasks_cache[env_id] = (now, resp.content)
        return _verbatim(resp)

    # -- operator surfaces ------------------------------------------------------

    @app.get("/host_status")
    async def host_status() -> dict[str, Any]:
        """Router-local resource snapshot (psutil-lite) + the additive
        ``fleet`` key. Same top-level shape family as a node's /host_status
        so a monitoring script can hit either."""
        vm = psutil.virtual_memory()
        load1, load5, load15 = os.getloadavg()
        async with fleet.lock:
            n_nodes = len(fleet.nodes)
            n_healthy = sum(n.healthy for n in fleet.nodes.values())
            n_instances = len(fleet.instance_map)
        return {
            "uptime_seconds": time.time() - fleet.started_at,
            "cpu": {"percent": psutil.cpu_percent(interval=None),
                    "load1": load1, "load5": load5, "load15": load15},
            "memory": {"total_gb": vm.total / (1 << 30),
                       "available_gb": vm.available / (1 << 30),
                       "percent": vm.percent},
            "fleet": {"nodes": n_nodes, "healthy": n_healthy, "instances": n_instances},
        }

    @app.get("/fleet/nodes")
    async def fleet_nodes() -> dict[str, Any]:
        """Full per-node detail — additive operator surface (no node analog)."""
        now = time.monotonic()
        async with fleet.lock:
            nodes = list(fleet.nodes.values())
        return {
            "nodes": [
                {
                    "name": n.name,
                    "base_url": n.base_url,
                    "healthy": n.healthy,
                    "last_seen_age_s": None if n.last_seen == 0.0 else now - n.last_seen,
                    "last_error": n.last_error,
                    "cooldown_remaining_s": max(0.0, n.cooldown_until - now),
                    "outstanding": n.outstanding,
                    "n_envs_cached": len(n.envs),
                    "build_info": n.build_info,
                }
                for n in nodes
            ]
        }

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        async with fleet.lock:
            n_nodes = len(fleet.nodes)
            n_healthy = sum(n.healthy for n in fleet.nodes.values())
            n_instances = len(fleet.instance_map)
        lines = [
            "# TYPE cua_lite_fleet_nodes gauge",
            f"cua_lite_fleet_nodes {n_nodes}",
            f"cua_lite_fleet_nodes_healthy {n_healthy}",
            "# TYPE cua_lite_fleet_instances_tracked gauge",
            f"cua_lite_fleet_instances_tracked {n_instances}",
        ]
        for key in sorted(fleet.counters):
            lines.append(f"# TYPE cua_lite_fleet_{key} counter")
            lines.append(f"cua_lite_fleet_{key} {fleet.counters[key]}")
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.api_route("/admin/{rest:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def admin_not_here(rest: str) -> Response:
        async with fleet.lock:
            urls = [n.base_url for n in fleet.nodes.values()]
        return JSONResponse(
            status_code=501,
            content={
                "detail": (
                    "admin endpoints are per-node in the fleet MVP — query each "
                    f"node directly: {urls}"
                ),
            },
        )

    return app
