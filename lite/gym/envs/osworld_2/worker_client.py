"""OSWorld Worker protocol client with idempotent action submission."""

from __future__ import annotations

import base64
import http.client
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

READ_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    ConnectionError,
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
    json.JSONDecodeError,
    UnicodeDecodeError,
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keep Worker credentials on the configured endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class WorkerClient:
    def __init__(self, endpoint, token, timeout=2100):
        url = urllib.parse.urlsplit(endpoint)
        if (
            url.scheme not in ("http", "https")
            or not url.netloc
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("Invalid worker endpoint")
        self.endpoint = endpoint.rstrip("/")
        self.token, self.timeout = token, timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def request(self, method, path, body=None, *, timeout=None):
        request = urllib.request.Request(
            self.endpoint + path,
            method=method,
            headers={"X-OSWorld-Session-Token": self.token, "Content-Type": "application/json"},
            data=json.dumps(body).encode() if body is not None else None,
        )
        deadline = None if timeout is None else time.monotonic() + timeout
        for attempt in range(3):
            remaining = 30 if deadline is None else deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Worker request exceeded its confirmation deadline")
            try:
                with self.opener.open(request, timeout=min(30, remaining)) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                if not (method == "GET" and error.code in (429, 502, 503, 504) and attempt < 2):
                    detail = error.read(2000).decode(errors="replace")
                    detail = detail.replace(self.token, "[redacted]")
                    raise RuntimeError(f"Worker HTTP {error.code}: {detail}") from None
            except READ_ERRORS:
                # A saved observation is immutable. Fetch the same ID again;
                # never repeat the GUI action just because its image was cut off.
                if method != "GET" or attempt == 2:
                    raise
            delay = 2**attempt
            if deadline is not None and time.monotonic() + delay >= deadline:
                raise TimeoutError("Worker request exceeded its confirmation deadline") from None
            time.sleep(delay)

    def operation(self, path, body):
        body = {"request_id": uuid.uuid4().hex, **body}
        # A dropped response must not duplicate an action. Retries reuse the ID.
        for attempt in range(3):
            try:
                submitted = self.request("POST", path, body)
                break
            except READ_ERRORS:
                if attempt == 2:
                    raise
                time.sleep(1)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                result = self.request("GET", "/operations/" + submitted["operation_id"])
            except READ_ERRORS:
                time.sleep(1)
                continue
            if result["status"] == "succeeded":
                return result["result"]
            if result["status"] == "failed":
                raise RuntimeError(result["error"])
            time.sleep(0.5)
        raise TimeoutError(
            f"Operation {submitted['operation_id']} timed out; inspect it before retrying"
        )

    def reset(self, task_id, run_id):
        return self.operation("/task/reset", {"task_id": task_id, "run_id": run_id})

    def observe(self, run_id, observation_id=None):
        query = {"run_id": run_id}
        if observation_id:
            query["observation_id"] = observation_id
        result = self.request("GET", "/observation?" + urllib.parse.urlencode(query))
        result["screenshot"] = base64.b64decode(result["screenshot"], validate=True)
        return result

    def task(self, run_id):
        """Read the coordinator's current task and phase without replaying setup."""
        return self.request("GET", "/task?" + urllib.parse.urlencode({"run_id": run_id}))

    def action(self, run_id, action, pause=0.2, *, phase_index=0):
        body = {"run_id": run_id, "phase_index": phase_index, "action": action, "pause": pause}
        return self.operation("/action", body)

    def ask_user(self, run_id, question, *, phase_index=0):
        return self.operation(
            "/ask_user", {"run_id": run_id, "phase_index": phase_index, "question": question}
        )

    def refresh(self, run_id, *, phase_index=0):
        return self.operation("/observation", {"run_id": run_id, "phase_index": phase_index})

    def evaluate(self, run_id, *, phase_index=0):
        return self.operation("/task/evaluate", {"run_id": run_id, "phase_index": phase_index})

    def advance(self, run_id, *, phase_index):
        return self.operation("/task/advance", {"run_id": run_id, "phase_index": phase_index})

    def close(self, run_id):
        return self.operation("/task/close", {"run_id": run_id})

    def wait_closed(self, run_id, worker_id, release, timeout=None):
        """Read a matching close receipt before reusing this Worker's episode slot.

        This method only reads health; it never submits cleanup or replays an
        episode operation. A server's detached DELETE is not a close receipt.
        """
        duration = float(self.timeout if timeout is None else timeout)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Close confirmation timeout must be finite and positive")
        deadline = time.monotonic() + duration
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Worker did not confirm cleanup for run {run_id}")
            health = self.request("GET", "/healthz", timeout=remaining)
            if health.get("worker_id") != worker_id or health.get("release") != release:
                raise RuntimeError("Close confirmation reached a different Worker or release")
            if "last_closed_run_id" not in health or "run_id" not in health:
                raise RuntimeError("Worker lacks close receipts; upgrade the Worker before reuse")
            if health["state"] == "broken":
                raise RuntimeError("Worker is broken; cleanup cannot be confirmed")
            if health["run_id"] not in (None, run_id) and (
                health["state"] != "idle" or health["busy"]
            ):
                raise RuntimeError("Worker is occupied by another run; cleanup cannot be confirmed")
            if (
                health["run_id"] == run_id
                and health["last_closed_run_id"] == run_id
                and health["state"] == "idle"
                and health["busy"] is False
            ):
                return health
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Worker did not confirm cleanup for run {run_id}")
            time.sleep(min(0.5, remaining))
