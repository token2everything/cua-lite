#!/opt/env/venv/bin/python
"""Fail unless the active local Chrome page has rendered application content."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

import websocket

_EXPRESSION = """JSON.stringify((() => {
  const root = document.getElementById('root');
  return {
    readyState: document.readyState,
    url: location.href,
    title: document.title,
    bodyChildren: document.body ? document.body.childElementCount : 0,
    rootPresent: Boolean(root),
    rootChildren: root ? root.childElementCount : 0,
    rootTextLength: root ? root.innerText.trim().length : 0
  };
})())"""


def _pages(port: int) -> list[dict]:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/json", timeout=2
    ) as response:
        return json.load(response)


def _inspect(page: dict) -> dict:
    ws = websocket.create_connection(
        page["webSocketDebuggerUrl"],
        timeout=3,
        suppress_origin=True,
    )
    try:
        ws.send(json.dumps({
            "id": 1,
            "method": "Runtime.evaluate",
            "params": {"expression": _EXPRESSION, "returnByValue": True},
        }))
        while True:
            message = json.loads(ws.recv())
            if message.get("id") != 1:
                continue
            result = message.get("result", {}).get("result", {})
            if result.get("type") != "string":
                raise RuntimeError(f"unexpected CDP result: {message}")
            return json.loads(result["value"])
    finally:
        ws.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9222)
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout
    last_error = "Chrome DevTools endpoint unavailable"

    while time.monotonic() < deadline:
        try:
            pages = [
                page for page in _pages(args.port)
                if page.get("type") == "page"
                and page.get("url", "").startswith(("http://127.0.0.1:", "http://localhost:"))
            ]
            if not pages:
                last_error = "no local application page found"
            else:
                states = [_inspect(page) for page in pages]
                unhealthy = [
                    state for state in states
                    if state["readyState"] not in {"interactive", "complete"}
                    or state["bodyChildren"] == 0
                    or (state["rootPresent"] and state["rootChildren"] == 0)
                ]
                if not unhealthy:
                    print(json.dumps(states, sort_keys=True))
                    return 0
                last_error = (
                    "page not rendered: "
                    f"{json.dumps(unhealthy, sort_keys=True)}"
                )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.5)

    print(last_error)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
