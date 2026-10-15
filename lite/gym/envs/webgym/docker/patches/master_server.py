"""VENDORED + PATCHED OmniBoxes master server (cua-lite).

Base: omniboxes/master/server.py @ pinned commit 574a205e (see Dockerfile).

PATCHES:
  * read ``WEBGYM_API_KEY`` from the container environment;
  * preserve the screenshot ``cursor`` query parameter and forward it to the
    worker node. Without this, host ``cursor=False`` is dropped at the master and
    screenshots always render with the default cursor setting downstream;
  * preserve the structured ``instance_dead`` flag on a failed ``/execute``
    (hop 3, the LAST hop — see ``_execute_error_body``). ``playwright_instance.py``
    reports a dead browser/page/context as
    ``{"detail": {"message": ..., "instance_dead": true}}``; ``node/server.py``
    forwards that verbatim; upstream's master then re-flattened EVERY non-200 into
    ``{"status": "error", "message": f"...: {response.text}"}``, destroying the
    typed fact and leaving the cua-lite host substring-matching "has been closed".
    The flag is now lifted to the TOP LEVEL of the master's error body.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import requests
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.responses import JSONResponse
from omniboxes.master.node_manager import NodeManager, NodeRegistration


async def get_api_key(x_api_key: str = Header(None, alias="x-api-key")):
    """Dependency to verify the API key in the request header."""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key is missing",
        )
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key.",
        )
    return x_api_key


parser = argparse.ArgumentParser(description="OmniBox Host")
parser.add_argument("--port", type=int, default=7000, help="Port to run the server on")
parser.add_argument("--nodes", type=str, nargs="+", default=[], help="List of node URLs to register")
parser.add_argument("--workers", type=int, default=1, help="Number of worker processes to run")
parser.add_argument("--redis-registry", action="store_true", help="Enable Redis-based node discovery")
parser.add_argument("--redis-host", type=str, default="localhost", help="Redis host for node registry")
parser.add_argument("--redis-port", type=int, default=6379, help="Redis port for node registry")
args = parser.parse_args()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("omnibox-master")

redis_registry = None
if args.redis_registry:
    from omniboxes.common.redis_registry import RedisRegistry

    logger.info("Enabling Redis-based node discovery: %s:%s", args.redis_host, args.redis_port)
    try:
        redis_registry = RedisRegistry(
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            registry_db=1,
            logger=logger,
        )
        logger.info("Redis registry initialized")
    except Exception as e:
        logger.error("Failed to initialize Redis registry: %s", e)
        redis_registry = None


API_KEY = os.environ.get("WEBGYM_API_KEY", "default_key")
NODE_API_KEY = API_KEY

node_manager = NodeManager(
    api_key=NODE_API_KEY,
    logger=logger,
    redis_registry=redis_registry,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    for node_url in args.nodes:
        await node_manager.register_node(NodeRegistration(url=node_url))
    tasks = [
        asyncio.create_task(node_manager.update_statuses_worker()),
        asyncio.create_task(node_manager.update_nodes()),
    ]
    yield
    for task in tasks:
        task.cancel()


app = FastAPI(
    title="OmniBox Master Node",
    description="Manages redirection of instance operations to worker nodes",
    lifespan=lifespan,
)


# cua-lite PATCH: wire name of the structured "this instance's browser is gone"
# flag. Defined as ``INSTANCE_DEAD_KEY`` in
# ``omniboxes/node/instances/playwright_instance.py``; repeated as a literal here
# on purpose — importing that module would drag playwright into the master
# process, which only proxies.
_INSTANCE_DEAD_KEY = "instance_dead"


def _execute_error_body(response: requests.Response) -> Dict[str, Any]:
    """Master-side error body for a failed ``/execute``, flag preserved.

    The node forwards the instance's structured 500 body verbatim, i.e.
    ``{"detail": {"message": ..., "instance_dead": true}}``. Upstream collapsed
    that into a single ``message`` string, so the only way for the host to learn
    the browser had died was to pattern-match prose. Lift the flag to the top
    level instead; ``message`` still carries the text, for logs only.
    """
    body: Dict[str, Any] = {
        "status": "error",
        "message": f"Failed to execute command: {response.text}",
        _INSTANCE_DEAD_KEY: False,
    }
    try:
        payload = response.json()
    except ValueError:
        # Non-JSON body (e.g. a proxy/uvicorn error page): nothing to preserve,
        # and "not known dead" is the honest answer.
        return body
    # Accept the flag at the top level too, so a future producer that stops
    # nesting under FastAPI's ``detail`` envelope keeps working.
    for source in (payload, payload.get("detail") if isinstance(payload, dict) else None):
        if isinstance(source, dict) and source.get(_INSTANCE_DEAD_KEY):
            body[_INSTANCE_DEAD_KEY] = True
    return body


@app.post("/get")
async def create_instance(lifetime_mins: int = 60, api_key: str = Depends(get_api_key)):
    """Get an available new instance from the least occupied node."""
    node = node_manager.get_best_node()
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No available nodes with capacity to create new instance",
        )

    data = requests.post(
        f"{node.url}/get",
        params={"lifetime_mins": lifetime_mins},
        headers={"x-api-key": NODE_API_KEY},
    ).json()
    if "instance_id" in data:
        await node_manager.lease_instance(node.hash, data["instance_id"])
        return {
            "instance_id": data["instance_id"],
            "node": node.hash,
        }
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="No available nodes with capacity to create new instance",
    )


@app.post("/reset")
async def reset(instance_id: str, node: str, api_key: str = Depends(get_api_key)):
    """Reset an existing instance by delegating to its worker node."""
    node_info = node_manager.get_node(node)
    if node_info is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Node {node} is not found",
        )
    response = requests.post(
        f"{node_info.url}/reset",
        params={"instance_id": instance_id},
        headers={"x-api-key": NODE_API_KEY},
    )
    return JSONResponse(content=response.json(), status_code=response.status_code)


@app.get("/probe")
async def probe(instance_id: str, node: str, api_key: str = Depends(get_api_key)):
    """Probe the instance by delegating to its worker node."""
    node_info = node_manager.get_node(node)
    if node_info is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Node {node} is not found",
        )
    response = requests.get(
        f"{node_info.url}/probe",
        params={"instance_id": instance_id},
        headers={"x-api-key": NODE_API_KEY},
    )
    return JSONResponse(content=response.json(), status_code=response.status_code)


@app.get("/screenshot")
async def screenshot(
    instance_id: str,
    node: str,
    interaction_mode: str = "set_of_marks",
    cursor: bool = True,
    api_key: str = Depends(get_api_key),
):
    """Take a screenshot by delegating to the worker node, preserving cursor."""
    node_info = node_manager.get_node(node)
    if node_info is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Node {node} is not found",
        )
    response = requests.get(
        f"{node_info.url}/screenshot",
        params={
            "instance_id": instance_id,
            "interaction_mode": interaction_mode,
            "cursor": "1" if cursor else "0",
        },
        headers={"x-api-key": NODE_API_KEY},
    )
    if response.status_code == 200:
        return Response(content=response.content, media_type="image/png")

    return JSONResponse(
        content={"status": "error", "message": f"Failed to get screenshot: {response.text}"},
        status_code=response.status_code,
    )


@app.post("/execute")
async def execute(command_data: Dict[str, Any], api_key: str = Depends(get_api_key)):
    """Forward execute command to the worker node for the specified instance."""
    node = command_data.pop("node")
    instance_id = command_data.pop("instance_id")
    node_info = node_manager.get_node(node)
    if node_info is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Node {node} is not found",
        )
    response = requests.post(
        f"{node_info.url}/execute",
        params={"instance_id": instance_id},
        json=command_data,
        headers={"x-api-key": NODE_API_KEY},
    )
    if response.status_code == 200:
        return JSONResponse(content=response.json(), status_code=response.status_code)

    # PATCH: keep the node's structured instance_dead flag (do not stringify).
    return JSONResponse(
        content=_execute_error_body(response),
        status_code=response.status_code,
    )


@app.get("/metadata")
async def metadata(instance_id: str, node: str, api_key: str = Depends(get_api_key)):
    node_info = node_manager.get_node(node)
    if node_info is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Node {node} is not found",
        )
    response = requests.get(
        f"{node_info.url}/metadata",
        params={"instance_id": instance_id},
        headers={"x-api-key": NODE_API_KEY},
    )
    if response.status_code == 200:
        return Response(content=response.content, media_type="image/png")

    return JSONResponse(
        content={"status": "error", "message": f"Failed to get metadata: {response.text}"},
        status_code=response.status_code,
    )


@app.get("/info")
def get_info(api_key: str = Depends(get_api_key)):
    node_info = node_manager.node_info()
    return JSONResponse(
        content={
            "nodes": [
                {
                    "url": node.url,
                    "hash": node.hash,
                    "healthy": node.healthy,
                    "capacity": node.capacity,
                    "available": node.available,
                    "instances": node.instances,
                }
                for node in node_info.values()
            ]
        },
        status_code=status.HTTP_200_OK,
    )


if __name__ == "__main__":
    uvicorn.run(
        "omniboxes.master.server:app",
        host="0.0.0.0",
        port=args.port,
        workers=args.workers,
    )
