"""Shared pickle-over-HTTP RPC client for in-container env-server proxies, plus
the ONE retry question every hop in the tree asks (:func:`may_reissue`).

Run: not directly. Two independent clients live here:

* :class:`RemoteRPC` (pickle mode) — imported by the android envs' host-side
  proxies (``androidworld`` / ``androidlab`` ``main.py``), which hand the
  upstream env/task code a ``_Remote*`` object that redirects every method
  call to a single ``POST`` on the in-container ``docker/server.py``.
* :func:`json_rpc` (JSON mode) — imported by ``osworld`` / ``osworld_2``
  ``main.py`` to talk to their in-container eval server.

Bodies and responses are **pickled** Python objects so protobuf / numpy /
android XML-judge structures survive the boundary with full fidelity; the
response Content-Type selects the decoder (``octet-stream`` → unpickle, else
JSON — healthz / status endpoints answer JSON). The hot path is one request →
one unpickle, no JSON-schema bookkeeping.

The two android envs' ``_RemoteRPC`` subclasses differ only in the ``retries``
constructor arg — androidworld passes ``retries=1`` (one retry on a transient
connection error), androidlab ``retries=0`` — so the body stays
single-sourced. The ``HTTPError`` message is NOT parameterized: :meth:`post`
raises one fixed ``server {path} returned {code}: {detail}`` for both.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
import urllib.error
import urllib.request
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def is_transport_error(exc: BaseException) -> bool:
    """Did the TRANSPORT fail — the socket broke, or never opened?

    TYPED ONLY; no message and no class NAME is read. Every producer stamps one
    of these at the boundary it owns (``json_rpc`` converts ``requests``' two
    transport families; httpx and urllib already type theirs).

    Excluded on purpose: an ANSWERED request (a non-2xx wrapped as
    ``RuntimeError``) and every timeout whose peer may still be working
    (``TimeoutError``, ``httpx.ReadTimeout`` / ``WriteTimeout`` /
    ``PoolTimeout``) — re-issuing those stacks a second copy of an op that is
    still running. ``httpx.ConnectTimeout`` is the one timeout that is IN: the
    connect phase never completed, so no request bytes were written.
    """
    return isinstance(exc, (
        ConnectionError,            # builtin — the typed raise every producer uses
        urllib.error.URLError,      # RemoteRPC (android envs' pickle RPC)
        httpx.NetworkError,         # host <-> env-server hop
        httpx.RemoteProtocolError,  # ... incl. a keepalive RST mid-response
        httpx.ConnectTimeout,       # connect phase never completed (see above)
    ))


#: Transport failures whose RAISER asserts the request never went out: the
#: connect phase is the only phase that can prove the negative. Read as a type,
#: never re-derived from a message.
_NEVER_REACHED = (
    ConnectionRefusedError, httpx.ConnectError, httpx.ConnectTimeout,
)


def reached_worker(exc: BaseException) -> bool:
    """Might the peer have STARTED executing this call? **Unknown counts as
    yes** — the polarity is the whole safety property.

    Only the connect phase proves the negative, and only the raiser knows it: a
    refused / never-opened connection put no bytes on the wire, while a reset or
    a cut response means the request WAS delivered. Anything else (a bare
    ``URLError``; a bare builtin ``ConnectionError``) has not been answered by
    anybody, so it must not license replaying a non-idempotent op.
    """
    if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, BaseException):
        # urllib puts the original socket error in ``.reason`` — a field the
        # raiser set, not a message to parse.
        exc = exc.reason
    return not isinstance(exc, _NEVER_REACHED)


def may_reissue(exc: BaseException, *, replay_safe: bool) -> bool:
    """May this exact call be sent again? — the ONE question every call-level
    retry loop asks, on both hops (host↔env-server, env-server↔container).

    Two inputs, both set by someone who knows rather than guessed from a message:

    * ``replay_safe`` — the caller's idempotence knowledge. When True the
      reached-worker question has no consequence (``/reset`` re-establishes the
      same initial state, so a replay costs at most time).
    * :func:`reached_worker` — the raiser's phase knowledge, load-bearing
      exactly when ``replay_safe`` is False. A connect-refused proves nothing
      executed, so re-issuing even a ``/step`` is safe; a reset or a cut
      response proves the opposite and is never replayed.
    """
    return is_transport_error(exc) and (replay_safe or not reached_worker(exc))


class RemoteRPC:
    """Tiny HTTP-RPC client to an in-container env-server.

    Args:
        base_url: Container env-server base URL (trailing slash stripped).
        timeout: Per-request urlopen timeout in seconds.
        retries: Extra attempts on a re-issuable transport failure (0 = single
            attempt, the androidlab default; 1 = androidworld's one retry).
            ``HTTPError`` is never retried — those are application-level
            rejects (e.g. 409) handled at the call site.
        retry_sleep: Seconds slept between a transient failure and the retry.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 180.0,
        *,
        retries: int = 0,
        retry_sleep: float = 2.0,
    ):
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._retries = retries
        self._retry_sleep = retry_sleep

    def post(self, path: str, body: Any = None) -> Any:
        """POST a pickled ``body``; unpickle (octet-stream) or JSON-parse the reply.

        On HTTPError raises ``RuntimeError(f"server {path} returned
        {code}: {detail}")``. Otherwise the retry decision is
        :func:`may_reissue`'s, up to ``retries`` times (logging +
        ``retry_sleep`` between attempts); the final failure propagates so the
        caller (env-server) can map it to 500.

        ``replay_safe=False`` is not a parameter: this client hands the upstream
        android env/task code a proxy that redirects EVERY method call —
        including the ones that press buttons — to one ``POST``, so the hop has
        no idempotence knowledge to thread and must take the conservative
        answer. It therefore retries only what the raiser PROVED never reached
        the container.
        """
        url = f"{self._base}{path}"
        data = pickle.dumps(body) if body is not None else b""
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        for attempt in range(self._retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    ct = resp.getheader("Content-Type", "")
                    body_bytes = resp.read()
                    if "octet-stream" in ct:
                        return pickle.loads(body_bytes)
                    return json.loads(body_bytes.decode("utf-8")) if body_bytes else None
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"server {path} returned {e.code}: {detail}"
                ) from e
            except Exception as e:
                if attempt >= self._retries or not may_reissue(e, replay_safe=False):
                    raise
                logger.warning(
                    "RPC %s transient %s; retrying once after %.0f s",
                    path, type(e).__name__, self._retry_sleep,
                )
                time.sleep(self._retry_sleep)


def _connect_phase_failed(exc: Exception) -> bool:
    """Did this ``requests.ConnectionError`` fail BEFORE any request bytes went
    out? Read off urllib3's typed ``.reason``; no message is parsed.

    ``requests.ConnectionError`` is the one transport family in the tree that
    spans both phases, so it is the one a producer must not flatten. ``requests``
    keeps the phase on the object (``adapters.py``): the connect phase arrives as
    ``MaxRetryError`` whose ``.reason`` is a
    ``urllib3.exceptions.ConnectTimeoutError`` — the base of
    ``NewConnectionError`` (refused connect) and ``NameResolutionError`` (DNS),
    and the type of a connect timeout itself. A failure once the socket was up
    arrives as ``ProtocolError`` (``RemoteDisconnected``) or a bare ``OSError``.

    Polarity matches :func:`reached_worker`: only the ``ConnectTimeoutError``
    branch PROVES the negative, so everything else answers False and is treated
    as may-have-executed.
    """
    import urllib3  # deferred with requests, which hard-depends on it

    reason = exc.args[0] if exc.args else None
    return isinstance(reason, urllib3.exceptions.MaxRetryError) and isinstance(
        reason.reason, urllib3.exceptions.ConnectTimeoutError,
    )


def json_rpc(
    base_url: str,
    path: str,
    body: dict | None = None,
    *,
    timeout: float,
    label: str,
    attempts: int = 3,
) -> dict:
    """JSON-over-HTTP RPC to an in-container eval server — the ONE
    implementation of the osworld/osworld_2 client (they were byte-identical
    copies with only the log label differing).

    Retries ONLY connect-level failures (the api port briefly unreachable — a
    load blip or a qemu-docker DNAT re-assert window), because those provably
    never reached the worker no matter what the op was. A non-200 is a real
    server error and propagates immediately (no retry); a mid-op ReadTimeout
    also propagates (the container is torn down + respawned on failure).
    ``timeout`` is the READ timeout; connect is fixed at 10 s.

    **Which arm a failure takes is decided by phase, not by family.** This layer
    does not know which op it is carrying, so it cannot know whether replaying a
    possibly-executed call is safe. It leaves every failure TYPED, and the type
    answers :func:`reached_worker` so the hop that holds ``replay_safe`` decides:

    * connect phase never completed (:func:`_connect_phase_failed`) → retried up
      to ``attempts`` times, then ``ConnectionRefusedError``: nothing ran, so
      even a ``/step`` may be re-issued;
    * the socket was up and then broke — ``RemoteDisconnected`` after the request
      was written, or a mid-response ``ChunkedEncodingError`` →
      ``ConnectionResetError`` (a ``ConnectionError`` subclass, so every
      ``except ConnectionError`` site is unchanged), raised on the FIRST
      occurrence with no retry: bytes were on the wire, so the op MAY have run.

    Retrying the second family is not merely wasteful, it is unsound: the loop
    would replay a possibly-executed ``/step`` up to ``attempts`` times and then
    hand the outer hop a never-reached claim, licensing it to send the op again.
    """
    import requests  # deferred: only the JSON-mode envs carry the dep on this path

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            r = requests.post(f"{base_url}{path}", json=body or {}, timeout=(10, timeout))
            if r.status_code != 200:
                raise RuntimeError(f"{label} server {path} → HTTP {r.status_code}: {r.text[:800]}")
            return r.json()
        except requests.ConnectionError as e:
            if not _connect_phase_failed(e):
                raise ConnectionResetError(
                    f"{label} server {path} lost the connection after the "
                    f"request was sent: {e}"
                ) from e
            last = e
            time.sleep(1.0 * (attempt + 1))
        except requests.exceptions.ChunkedEncodingError as e:
            raise ConnectionResetError(
                f"{label} server {path} cut mid-response: {e}"
            ) from e
    # Only connect-phase failures reach here — the arms above diverted the rest —
    # so the type is a claim this function can actually make.
    # ``ConnectionRefusedError`` (not the bare ``ConnectionError``) so
    # :func:`reached_worker` stays a plain isinstance tuple: the three members
    # that get here — refused, DNS failure, connect timeout — share the property
    # being encoded, and the message carries which one it was.
    raise ConnectionRefusedError(
        f"{label} server {path} unreachable after {attempts} tries: {last}"
    ) from last
