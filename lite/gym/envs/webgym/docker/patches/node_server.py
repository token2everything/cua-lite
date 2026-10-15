"""VENDORED + PATCHED OmniBoxes node server (cua-lite).

Base: omniboxes/node/server.py @ pinned commit 574a205e (see docker/Dockerfile).
The Dockerfile clones upstream @ that SHA and ``cp``s this file over
``/opt/omniboxes/omniboxes/node/server.py`` before ``pip install`` — so there is
no dependency on any cua-lite fork and the delta is reviewable in one place.

PATCH — lease-TTL reaper (fixes instance leaks on client death):
  Upstream OmniBoxes lifecycle is "100% manual": ``/get`` moves a port
  ``available → in_use`` and ONLY ``/reset`` returns it; ``lifetime_mins`` is
  ignored and nothing reaps a stale ``in_use`` lease. So a client that dies /
  is ``kill -9``'d without calling ``/reset`` leaks its instance permanently
  (until pool restart) → the 32/64-slot pool exhausts after a few killed runs.

  This mirrors the cua-lite env-server's design (last-activity timestamp +
  ``idle_ttl`` reaper, server.py ``_reap_idle``): every lease records a
  ``lease_active`` timestamp (Redis hash), each interaction refreshes it, and a
  background task reclaims any ``in_use`` lease idle past ``OMNIBOXES_LEASE_TTL_S``
  (default 600 s) — busy tasks keep renewing so they're never reaped, a dead
  client stops renewing and is reclaimed within the TTL. Self-healing in both
  direct and env-server mode.

INVARIANT (hop 2 of the ``instance_dead`` chain — do NOT stringify):
  ``playwright_instance.py`` reports a dead browser/page/context as a STRUCTURED
  ``{"detail": {"message": ..., "instance_dead": true}}`` 500 body so the host
  never has to substring-match an error message. This server is the middle hop:
  ``/execute`` must forward that body **as JSON** (it does —
  ``JSONResponse(content=response.json(), status_code=...)``). Re-wrapping it as
  ``detail=response.text`` (the shape the sibling error branches use for network
  failures) would flatten the flag back into prose and silently break the host's
  fail-fast on a dead instance.
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse, Response
import uvicorn
from typing import Dict, Any
from contextlib import asynccontextmanager
import argparse
import asyncio
import os
import socket
import time
from omniboxes.node.instances.playwright_instance import PlaywrightInstance
from omniboxes.node.logging_utils import default_logger
from omniboxes.node.utils import GetApiKey
from fastapi.responses import JSONResponse
from pathlib import Path
import redis
import httpx

parser = argparse.ArgumentParser(description="OmniBox Host")
parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
parser.add_argument('--lifetime_max', type=int, default=60, help="Maximum lifetime of an instance in minutes")
parser.add_argument('--redis_port', type=int, default=6379, help="Port for Redis connection")
parser.add_argument('--redis_host', type=str, default='localhost', help="Host for Redis connection")
parser.add_argument('--workers', type=int, default=1, help="Number of worker processes to run")
args = parser.parse_args()

API_KEY = os.environ.get("WEBGYM_API_KEY", "default_key")

logger = default_logger()

INSTANCE_API_KEY = API_KEY    # todo: make separate

r = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)

# ─── cua-lite PATCH: lease-TTL reaper config ────────────────────────────────
# Reclaim an ``in_use`` lease whose last activity is older than this (a dead
# client never refreshes). Mirrors the env-server idle_ttl (default 600s). Must
# exceed the longest gap between a live task's interactions (LLM step latency).
_LEASE_TTL_S = float(os.environ.get("OMNIBOXES_LEASE_TTL_S", "600"))
_REAP_INTERVAL_S = max(5.0, min(60.0, _LEASE_TTL_S / 4.0))
_LEASE_ACTIVE = "lease_active"  # Redis hash: instance_id -> last-activity epoch
_REAP_LOCK = "lease_reaper_lock"          # single-reaper election key
_REAP_TOKEN = f"{socket.gethostname()}:{os.getpid()}"
# Refresh-if-owner as ONE atomic step (CAS). A non-atomic get()+set() would let
# a holder that loses the lock at expiry (another worker grabs it via NX in the
# gap) clobber it back with an unconditional set() — running two reapers in the
# same cycle. The Lua compare-and-set only refreshes when we still own the lock.
_REAP_REFRESH_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('set', KEYS[1], ARGV[1], 'EX', ARGV[2]) else return nil end"
)


def _touch_lease(instance_id: str) -> None:
    """Refresh a lease's last-activity timestamp (the 'heartbeat')."""
    try:
        r.hset(_LEASE_ACTIVE, instance_id, time.time())
    except redis.RedisError:
        pass


def _own_reaper(ttl: int) -> bool:
    """Single-reaper election across uvicorn workers.

    The node runs ``--workers N`` (32 here), so uvicorn forks N processes and
    runs ``lifespan`` — hence ``_lease_reaper`` — once PER worker. N reapers
    scanning the same Redis would race on the non-atomic release sequence. A
    Redis lock keeps exactly one worker reaping: the holder refreshes the lock
    each cycle, and if it dies the lock expires so another worker takes over
    within ``ttl``.
    """
    try:
        if r.set(_REAP_LOCK, _REAP_TOKEN, nx=True, ex=ttl):
            return True  # lock was free → we won it atomically
        # Otherwise refresh ONLY if we still own it, as one atomic CAS (a
        # get()+set() race would let us clobber a lock another worker grabbed in
        # the gap, running two reapers in one cycle).
        return bool(r.eval(_REAP_REFRESH_LUA, 1, _REAP_LOCK, _REAP_TOKEN, ttl))
    except redis.RedisError:
        return False


async def _lease_reaper() -> None:
    """Reclaim ``in_use`` leases idle past ``_LEASE_TTL_S`` (env-server _reap_idle
    analog). Resets the instance, returns its port to ``available``.

    NOTE: uses the synchronous redis client on the event loop — acceptable here
    because exactly one worker (see ``_own_reaper``) runs this every
    ``_REAP_INTERVAL_S`` over at most a pool-sized set, so the blocking is brief.
    """
    lock_ttl = int(_REAP_INTERVAL_S * 3)
    while True:
        await asyncio.sleep(_REAP_INTERVAL_S)
        if not _own_reaper(lock_ttl):
            continue  # another worker owns the reaper this cycle
        try:
            now = time.time()
            for m in r.smembers('in_use'):
                iid = m.decode('utf-8')
                ts = r.hget(_LEASE_ACTIVE, iid)
                if ts is None:
                    # Untracked lease (pre-patch or a /get↔reaper race): grant a
                    # grace window rather than reap immediately.
                    _touch_lease(iid)
                    continue
                if now - float(ts) <= _LEASE_TTL_S:
                    continue
                # Orphaned lease → reset the instance, then release the port.
                try:
                    inst, port = iid.split(':')
                except ValueError:
                    r.srem('in_use', iid); r.hdel(_LEASE_ACTIVE, iid)
                    continue
                # Only release if the reset SUCCEEDED — never return a dirty
                # (un-reset) instance to the pool. Mirrors the /reset handler,
                # which keeps a failed-reset instance in_use rather than leaking
                # stale page/cookie state to the next client.
                ok = False
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(
                            f"http://127.0.0.1:{port}/reset",
                            params={'instance_id': inst}, timeout=30.0,
                            headers={"x-api-key": INSTANCE_API_KEY},
                        )
                    ok = resp.status_code == 200
                    if not ok:
                        logger.warning(
                            f"lease reaper: reset for {iid} returned "
                            f"{resp.status_code}; keeping in_use, retry next cycle"
                        )
                except Exception as e:
                    logger.warning(
                        f"lease reaper: reset failed for {iid}: {e}; "
                        f"keeping in_use, retry next cycle"
                    )
                if not ok:
                    continue
                # CAS: skip release if the lease was refreshed while we reset
                # (defends against a re-lease/heartbeat racing the reset).
                if r.hget(_LEASE_ACTIVE, iid) != ts:
                    continue
                r.srem('in_use', iid)
                r.sadd('available', port)
                r.hdel(_LEASE_ACTIVE, iid)
                logger.warning(
                    f"lease reaper: reclaimed orphaned lease {iid} "
                    f"(idle > {_LEASE_TTL_S:.0f}s)"
                )
        except redis.RedisError as e:
            logger.warning(f"lease reaper: redis error: {e}")
        except Exception as e:
            logger.warning(f"lease reaper: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    reaper = asyncio.create_task(_lease_reaper())
    yield
    reaper.cancel()

app = FastAPI(lifespan=lifespan)

@app.post("/get")
async def get_instance(lifetime_mins: int = 60, api_key: str = Depends(GetApiKey(API_KEY))):
    """Get an available instance from the pool"""
    if lifetime_mins < 0 or lifetime_mins > args.lifetime_max:
        raise HTTPException(status_code=400, detail=f"Invalid lifetime_mins: {lifetime_mins}. Must be between 0 and {args.lifetime_max} minutes.")
    port = r.spop('available')
    if not port:
        raise HTTPException(status_code=503, detail="No instances available")
    async with httpx.AsyncClient() as client:
        try:
            port = port.decode('utf-8')
            response = await client.post(f"http://127.0.0.1:{port}/get", timeout=None, headers={"x-api-key": INSTANCE_API_KEY})
            instance_id = f'{response.json()["instance_id"]}:{port}'

            # Track the in-use instance + start its lease clock (PATCH: set the
            # lease timestamp BEFORE adding to in_use so the reaper never sees an
            # untracked member during the window).
            _touch_lease(instance_id)
            r.sadd('in_use', instance_id)

            return JSONResponse(content = {'instance_id': instance_id}, status_code=response.status_code)
        except httpx.RequestError as e:
            # If there was an error, put the port back to available
            r.sadd('available', port)
            raise HTTPException(status_code=500, detail=f"Error leasing the instance")


@app.post("/reset")
async def reset_instance(instance_id: str, api_key: str = Depends(GetApiKey(API_KEY))):
    """Reset an instance to its initial state and make it available again"""
    # Check if this instance is actually in use before resetting
    # This prevents race conditions where an instance has already been reallocated
    full_instance_id = instance_id
    in_use_instances = [m.decode('utf-8') for m in r.smembers('in_use')]
    if full_instance_id not in in_use_instances:
        logger.info(f"Instance {full_instance_id} not in use (already released or reallocated), skipping reset")
        raise HTTPException(status_code=400, detail="Instance not in use or already released")

    instance_id, port = instance_id.split(':')
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"http://127.0.0.1:{port}/reset", params = {'instance_id': instance_id}, timeout=None, headers={"x-api-key": INSTANCE_API_KEY})

            # Only remove from in_use if reset succeeded
            if response.status_code == 200:
                full_instance_id = f'{instance_id}:{port}'
                r.srem('in_use', full_instance_id)
                r.hdel(_LEASE_ACTIVE, full_instance_id)  # PATCH: clear lease clock
                logger.info(f"Successfully reset and released instance {full_instance_id}")
            else:
                logger.warning(f"Instance reset failed with status {response.status_code}, keeping in 'in_use': {response.text}")

            return JSONResponse(content = response.json(), status_code=response.status_code)
        except httpx.RequestError as e:
            logger.error(f"Network error resetting instance {instance_id}:{port}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error resetting the instance")

@app.get("/probe")
async def probe_instance(instance_id: str, api_key: str = Depends(GetApiKey(API_KEY))):
    """Forward probe request to the Flask server in the specified instance"""
    _touch_lease(instance_id)  # PATCH: interaction refreshes the lease
    instance_id, port = instance_id.split(':')
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"http://127.0.0.1:{port}/probe", params = {'instance_id': instance_id}, timeout=None, headers={"x-api-key": INSTANCE_API_KEY})
            return JSONResponse(content = response.json(), status_code=response.status_code)
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Error probing the instance")

@app.get("/screenshot")
async def get_instance_screenshot(
    instance_id: str,
    interaction_mode: str = "set_of_marks",
    cursor: bool = True,
    api_key: str = Depends(GetApiKey(API_KEY)),
):
    """Forward screenshot request to the Flask server in the specified instance"""
    _touch_lease(instance_id)  # PATCH: interaction refreshes the lease
    instance_id, port = instance_id.split(':')
    async with httpx.AsyncClient() as client:
        try:
            # Fetch the full content to avoid streaming errors
            proxied = await client.get(
                f"http://127.0.0.1:{port}/screenshot",
                params={
                    'instance_id': instance_id,
                    'interaction_mode': interaction_mode,
                    'cursor': '1' if cursor else '0',
                },
                timeout=None,
                headers={"x-api-key": INSTANCE_API_KEY}
            )
            return Response(content=proxied.content, media_type=proxied.headers.get("content-type"))
        except httpx.RequestError:
            raise HTTPException(status_code=500, detail="Error getting screenshot from instance")


@app.post("/execute")
async def execute_instance_command(instance_id: str, command_data: Dict[str, Any], api_key: str = Depends(GetApiKey(API_KEY))):
    """Forward execute command to the Flask server in the specified instance"""
    _touch_lease(instance_id)  # PATCH: interaction refreshes the lease
    instance_id, port = instance_id.split(':')
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"http://127.0.0.1:{port}/execute", params={'instance_id': instance_id}, json=command_data, timeout=None, headers={"x-api-key": INSTANCE_API_KEY})
            # PATCH invariant (see module docstring): forward the instance body
            # VERBATIM as JSON on every status, including 500 — this is the hop
            # that carries the structured ``instance_dead`` flag to the master.
            return JSONResponse(content = response.json(), status_code=response.status_code)
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Error executing command on instance")

@app.get("/info")
async def get_info(api_key: str = Depends(GetApiKey(API_KEY))):
    """Return information about instances"""
    available = r.scard('available')
    in_use_set = r.smembers('in_use')
    in_use = [member.decode('utf-8') for member in in_use_set]

    # Calculate total capacity as available + in_use
    capacity = available + len(in_use)

    return {
        "available": available,
        "capacity": capacity,
        "in_use": in_use
    }

@app.get("/metadata")
async def metadata(instance_id: str, api_key: str = Depends(GetApiKey(API_KEY))):
    _touch_lease(instance_id)  # PATCH: interaction refreshes the lease
    instance_id, port = instance_id.split(':')
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"http://127.0.0.1:{port}/metadata", params = {'instance_id': instance_id}, timeout=None, headers={"x-api-key": INSTANCE_API_KEY})
            return JSONResponse(content = response.json(), status_code=response.status_code)
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Error getting the instance metadata")


if __name__ == "__main__":
    uvicorn.run('omniboxes.node.server:app', host="0.0.0.0", port=args.port, log_level="info", loop="asyncio", workers=args.workers)
