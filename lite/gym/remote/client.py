"""HTTP client for a remote env server.

Drop-in replacement for a locally-instantiated :class:`LiteBaseEnv`:
forwards ``reset`` / ``step`` / ``close`` to an env server over HTTP.
Used when ``CUA_LITE_ENV_SERVER_URL`` is set (or ``env_server_url`` is
passed explicitly into ``gym.make``).

The wire format is defined in the sibling :mod:`lite.gym.remote.server`;
``scripts/serve_env.py`` is just a thin launcher for that module.

Run (smoke test, requires a running serve_env.py):
    CUA_LITE_ENV_SERVER_URL=http://localhost:30100 \\
    CUA_LITE_ENV_SERVER_TOKEN=xxx \\
    uv run python -c "
    import asyncio, lite.gym as gym
    async def main():
        env = gym.make('lite.osworld@osworld_chrome_030eeff7')
        r = await env.reset()
        print(r.text)
        await env.close()
    asyncio.run(main())
    "
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from typing import Any

import httpx

import lite.gym.remote.errors as _remote_errors  # noqa: F401  (registers remote error types)
from lite.core.metadata import LiteBaseMetadata, metadata_from_dict
from lite.core.tools.calls import RuntimeEnvAction
from lite.gym.base import LiteBaseEnv
from lite.gym.errors import (
    CapacityExhausted,
    EnvServerUnavailable,
    EnvUnavailable,
    LiteGymError,
    RemoteEnvError,
    RemoteWireProtocolError,
    lite_error_from_payload,
)
from lite.gym.remote.errors import (
    PAYLOAD_METADATA_MAX_VALUE_CHARS,
    action_payload_metadata,
)
from lite.gym.remote.frame import (
    BadMagic,
    BadVersion,
    TruncatedFrame,
    decode_reset_observation,
    decode_step_result,
)
from lite.gym.types import (
    LiteEnvObservation,
    LiteEnvStepResult,
)
from lite.gym.utils.backend.rpc import may_reissue

logger = logging.getLogger(__name__)

# httpx connection-pool sizing — tuned for 32+ concurrent rollouts (the
# ENV_CONCURRENCY ceiling used by GRPO recipes). The defaults
# (max_connections=100, max_keepalive_connections=20) force TCP
# teardown+reconnect churn on the long-tail of step/reset RPCs above 20
# keepalives, which is marginal on loopback (<1 ms/handshake) but
# meaningful for a remote env-server (~5–10 ms RTT × thousands of calls).
# Sized to comfortably exceed ENV_CONCURRENCY=32 with headroom for
# burst create/close traffic.
_HTTPX_MAX_CONNECTIONS = 200
_HTTPX_MAX_KEEPALIVE = 100

#: How long this pool is willing to hand out an idle keep-alive connection.
#:
#: **PUBLIC — half of a cross-process invariant.** The env-server's uvicorn
#: ``timeout_keep_alive`` MUST be strictly greater than this value, so that the
#: *client* always retires a pooled connection before the *server* closes it.
#: The server side derives its timeout from this constant
#: (``lite.gym.remote.alive.SERVER_KEEP_ALIVE_TIMEOUT_SEC``) so the two
#: cannot drift apart; ``tests/gym/remote/test_keep_alive_contract.py`` pins
#: the ordering.
#:
#: Why the ordering matters: uvicorn's ``timeout_keep_alive`` defaults to 5 s.
#: With that default the server closes idle connections this pool still
#: considers live, so any agent think-time above 5 s between two RPCs races the
#: close. httpcore usually notices the FIN (``has_expired()`` polls the socket
#: for readability) and dials a fresh connection, but when the close lands
#: *after* that check and *before* the response — which a busy single-threaded
#: uvicorn loop makes common, since the keep-alive callback is deferred behind
#: whatever else the loop is running — the request dies in
#: ``_receive_response_headers`` with ``httpx.ReadError``. ``/step`` is
#: deliberately not retried (see below), so that kills the episode.
HTTPX_KEEPALIVE_EXPIRY_SEC = 300.0


# Transport-drop retry budget for this hop, mirroring ``server.py``'s
# ``_OUTER_RETRY_ATTEMPTS`` / ``_OUTER_RETRY_BASE_DELAY_S`` for the hop below
# it: jittered exp backoff 2/4/8 s × ±50 %, ~14 s worst case across 4 attempts,
# which absorbs a keepalive RST storm without bloating httpx timeouts.
#
# Both hops ask ONE question — ``may_reissue(exc, replay_safe=...)`` — so the
# retried class of error is the same on both (a transport failure) and the
# per-operation part travels as the flag: ``/reset`` True, ``/step`` False. The
# drop this most often absorbs is rootless docker's slirp4netns forwarder
# RST'ing keepalive connections mid-response when its single thread saturates.
#
# Not env-var-tunable: the two knobs that used to set these were read under the
# SAME names in ``server.py`` for the other hop's loop, so tuning one silently
# moved both, and nothing in the repo ever set either.
_CLIENT_RETRY_ATTEMPTS = 4
_CLIENT_RETRY_BASE_DELAY_S = 2.0

# 503 / capacity-exhausted retries are bounded by a wall-clock deadline
# rather than an attempt count. The server's 503 carries an adaptive
# Retry-After per
# :func:`~lite.gym.remote.admission._docker_sema_retry_after_s` —
# which can be a few seconds during a brief spike or up to ~minutes
# under sustained docker_sema queue backpressure. Counting attempts
# was the wrong unit: 3 × (10 s hint) = 30 s of patience even when
# the queue clears in 4 minutes, so most clients gave up early and the
# rollout's group-redraw absorbed avoidable churn. A 5-minute deadline
# matches the worst-case queue clearance for a 64-burst
# ``ENV_CONCURRENCY`` host with the historical
# ``docker_create_concurrency=8`` default (host-derived since the
# capacity-module rev).
_CLIENT_503_DEADLINE_S = float(os.environ.get(
    "CUA_LITE_503_DEADLINE_S", "300.0"  # 5 minutes
))
_CLIENT_503_DEFAULT_RETRY_AFTER_S = 5.0  # fallback when header missing

# Eager-create (`POST /instances`) has its own retry loop because it's
# synchronous (runs from __init__) and the failure mode is different
# (admission queue full, not just capacity). Bounded by a wall-clock
# deadline so a permanently-full server doesn't hang gym.make forever.
_EAGER_CREATE_DEADLINE_S = float(os.environ.get(
    "CUA_LITE_EAGER_CREATE_DEADLINE_S", "600.0"  # 10 minutes
))


def _client_retry_delay_s(attempt_0_indexed: int) -> float:
    """Same exp-backoff + jitter formula as server.py's
    ``_retry_delay_s`` — keeping the formula identical means a load
    storm hitting both retry loops walks them apart at the same rate.
    """
    base = _CLIENT_RETRY_BASE_DELAY_S * (2 ** attempt_0_indexed)
    return base * (0.5 + random.random())


def _parse_retry_after(header: str | None) -> float:
    """Parse the server's ``Retry-After`` header. Our env-server always
    emits integer seconds (the only form our handler produces); we do
    NOT support RFC 7231's HTTP-date alternative because nothing
    upstream needs it. Defaults to :data:`_CLIENT_503_DEFAULT_RETRY_AFTER_S`
    on missing / malformed headers — the client still backs off rather
    than hammering the server.
    """
    if not header:
        return _CLIENT_503_DEFAULT_RETRY_AFTER_S
    try:
        return max(0.0, float(int(header)))
    except (ValueError, TypeError):
        return _CLIENT_503_DEFAULT_RETRY_AFTER_S


def _capacity_exhausted_from_response(
    response: httpx.Response,
    fallback_retry_after_s: float,
) -> CapacityExhausted:
    """Reconstruct the server's typed ``CapacityExhausted`` from a 503.

    Goes through the SAME registry reconstructor as every other typed body
    (``lite_error_from_payload`` → ``CapacityExhausted.from_payload``), so the
    field names live on the class and this module never re-derives them.

    The tail covers a 503 this deployment did not produce (a reverse proxy, an
    unrelated middleware): no ``error_type`` to rebuild from, and the header is
    the only Retry-After hint there is. It still returns ``CapacityExhausted``
    because a bare 503 IS the capacity fact and the caller's wait loop keys on
    the type.
    """
    try:
        payload = response.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        exc = lite_error_from_payload(payload)
        if isinstance(exc, CapacityExhausted):
            return exc
    return CapacityExhausted(
        what=response.text[:120] or "server returned 503 (no body)",
        retry_after_s=fallback_retry_after_s,
    )


def _env_unavailable_from_response(response: httpx.Response) -> EnvUnavailable:
    """Typed `EnvUnavailable` from an UNTYPED terminal 501 (older server / a node
    body forwarded verbatim by the fleet router). The typed path handles every
    body this server version emits; this is the version-skew fallback."""
    try:
        payload = response.json()
        what = payload.get("detail") or payload.get("what") or response.text[:200]
    except Exception:
        what = response.text[:200] or f"env-server returned {response.status_code}"
    return EnvUnavailable(what=str(what), status_code=response.status_code)


def _response_json(r: httpx.Response, op_name: str) -> Any:
    """Parse a 2xx response body, mapping an empty/truncated body to a typed
    typed :class:`RemoteEnvError` instead of a raw ``JSONDecodeError``.

    Observed failure mode under host saturation: the
    server's reset hit its 600s cap at the same moment the client's own
    timeout expired — the response write raced the cancellation and the
    client read an empty 200/500 body. Callers decide whether the operation is
    idempotent enough to retry after this classification."""
    try:
        return r.json()
    except Exception as e:
        raise RemoteEnvError(
            what=f"{op_name}: malformed/empty {r.status_code} response body ({e})",
            original_type="MalformedServerResponse",
        ) from e


#: Cap on the >=400 body excerpt written to the log. Enough for the typed
#: envelope's fields plus a sentence of ``what``; short enough that a server
#: returning an HTML error page or a megabyte of traceback cannot flood the log.
_ERROR_BODY_MAX = 400


def _error_body_summary(r: httpx.Response) -> str:
    """Bounded one-line rendering of a >=400 body, for the log.

    The typed envelope's ``error_type``/``phase``/``kind``/``returncode`` + ``what``
    are what a failed remote setup is debugged from, so log them before raising.
    This is still useful when the client can reconstruct the payload typed, and
    critical for unknown envelopes that fall through to ``raise_for_status``,
    which reports the status and URL but never the body.
    """
    try:
        payload = r.json()
    except Exception:
        return (r.text or "<empty body>")[:_ERROR_BODY_MAX]
    if not isinstance(payload, dict):
        return str(payload)[:_ERROR_BODY_MAX]
    tagged = " ".join(
        f"{key}={payload[key]!r}"
        for key in ("error_type", "phase", "kind", "returncode")
        if payload.get(key) is not None
    )
    what = str(payload.get("what") or payload.get("detail") or "")
    return f"{tagged} {what}".strip()[:_ERROR_BODY_MAX] or "<empty body>"


def _raise_if_step_body_not_json_serializable(actions: list[Any]) -> None:
    """Fail before httpx/json sees non-wire-safe model payloads.

    Remote mode must surface malformed model output as a typed, bounded
    ModelOutputError instead of leaking bytes/object reprs through a local
    TypeError. This is a transport preflight only; pairable Lite envelopes still
    go to the env so direct and remote ingress policy stays equivalent.
    """
    try:
        json.dumps({"actions": actions}, allow_nan=False)
        return
    except (TypeError, ValueError) as error:
        payload_metadata: dict[str, Any] = {
            "json_error_type": type(error).__name__,
            "json_error": str(error)[:PAYLOAD_METADATA_MAX_VALUE_CHARS],
        }
        for index, action in enumerate(actions):
            try:
                json.dumps(action, allow_nan=False)
            except (TypeError, ValueError):
                payload_metadata.update(
                    action_payload_metadata(index, action)
                )
                break
        raise _remote_errors.ModelOutputError(
            "malformed /step request body: client could not JSON-serialize "
            "body.actions",
            payload_metadata=payload_metadata,
        ) from error


def _log_error_response(r: httpx.Response, op_name: str) -> None:
    """Log every >=400 that is about to be raised, body included (bounded).

    Called at the TOP of the create and per-op >=400 call sites — above every branch
    that raises, including create's terminal 501 — so the body is on the record
    whether or not the client can reconstruct the error typed. 503 (capacity) is
    handled earlier and logs its own line, so this never fires on the routine warming
    path. It does NOT cover ``fetch_remote_tasks``, which raises its own 401/404/501
    and generic >=400 without routing through here."""
    logger.warning(
        "%s: env-server returned HTTP %d — %s",
        op_name, r.status_code, _error_body_summary(r),
    )


def _raise_typed_error_if_any(r: httpx.Response) -> None:
    """Reconstruct a typed error from a >=400 JSON body, if it carries one.

    ``error_type`` owns the failure class and the status is presentation, so
    registered ``{error_type, what}`` envelopes come back as their real classes
    via ``lite_error_from_payload``, status UNREAD.

    The one numeric branch below is the fallback for a body this client cannot
    read typed. An UNKNOWN ``error_type`` on a
    500 (the server's catch-all for a bare exception in env code) or a
    502 (the fleet router's proxy-hop synthesis for an owner-alive-but-
    failing forward) wraps as retryable :class:`RemoteEnvError` so the
    caller sees the original class name, not an opaque HTTPStatusError. It
    stays pinned to those two statuses: widening it would swallow the
    caller's ``501`` arm, which must stay TERMINAL.
    Non-JSON and non-object JSON bodies fall through silently — the caller's
    ``raise_for_status`` owns those."""
    try:
        payload = r.json()
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    exc = lite_error_from_payload(payload)
    if exc is not None:
        raise exc
    if r.status_code in (500, 502) and payload.get("error_type"):
        raise RemoteEnvError(
            what=payload.get("what", ""),
            original_type=payload["error_type"],
        )


def _decode_frame(
    r: httpx.Response,
    op_name: str,
    decode,
) -> LiteEnvObservation | LiteEnvStepResult:
    """Decode a 200 reset/step **binary frame**. The >=400 path is
    already resolved in ``_post_with_retry`` (typed error + ``raise_for_status``),
    so ``r`` here is a success response.

    Version-skew is **terminal** ("upgrade the client"), not a 300s retry-storm:
    a JSON 200 means an OLD (pre-frame) server, and ``BadMagic``/``BadVersion`` mean
    a foreign/newer wire. ``TruncatedFrame`` — an empty/short/cut body, the live
    dropped-response race — is classified separately so callers can replay only
    idempotent operations. The codec's typed contract is TOTAL, so this maps
    cleanly without re-parsing the frame here."""
    if r.headers.get("content-type", "").startswith("application/json"):
        raise RemoteWireProtocolError(
            f"{op_name}: server returned JSON, not a binary frame — "
            "client/server version skew; upgrade the client"
        )
    try:
        return decode(r.content)
    except (BadMagic, BadVersion) as e:
        raise RemoteWireProtocolError(
            f"{op_name}: unrecognized frame ({e}) — version skew; upgrade the client"
        ) from e
    except TruncatedFrame as e:
        raise RemoteEnvError(
            what=f"{op_name}: empty/truncated {r.status_code} frame ({e})",
            original_type="MalformedServerResponse",
        ) from e


def _decode_reset_frame(r: httpx.Response, op_name: str) -> LiteEnvObservation:
    return _decode_frame(  # type: ignore[return-value]
        r, op_name, decode_reset_observation
    )


def _decode_step_frame(r: httpx.Response, op_name: str) -> LiteEnvStepResult:
    return _decode_frame(r, op_name, decode_step_result)  # type: ignore[return-value]


class LiteEnvClient(LiteBaseEnv, allow_metadata_override=True):
    """HTTP client to a remote env server.

    Behaves as a :class:`LiteBaseEnv` from the caller's perspective. Eager:
    constructing the client synchronously ``POST``s ``/instances`` so metadata
    is authoritative before the first ``reset()``.

    Args:
        server_url: Env server base URL, e.g. ``http://env-node-1:30100``.
        env_key: Same format as ``gym.make`` (``"{env_id}@{task_id}"``).
        metadata: Optional registry-time metadata fallback. Used only if
            the eager POST /instances below is somehow skipped. The
            authoritative metadata that ``self.metadata`` returns comes
            from the server's create response (env_kwargs-shaped).
        token: Bearer token. Falls back to ``$CUA_LITE_ENV_SERVER_TOKEN``.
        session_id: Falls back to ``$SESSION_ID`` (matches the cleanup/session
            label used by ``scripts/train/slime/launch.sh``).
        timeout: httpx request timeout for every call. Default 600s — env
            ``reset()`` on osworld can take several minutes. ``gym.make``
            passes ``max(reset_timeout, step_timeout) + 60`` (660 s at the
            default ``reset_timeout``), which covers ONE server-side reset,
            not the server's ~4× retry budget for the same call: exceeding it
            raises :class:`~lite.gym.remote.errors.RemoteCallTimeout`, whose
            docstring is the contract for that asymmetry.
        **env_kwargs: Forwarded to the server's ``gym.make()``.

    Mental-model parity with direct mode: ``__init__`` synchronously POSTs
    ``/instances`` so that, by the time ``LiteEnvClient(...)`` returns, the
    server-side env has been constructed and ``self.metadata`` reflects the
    LIVE env_kwargs-shaped metadata (extra_tools / valid_actions / etc).
    This matches direct mode where ``gym.make()`` returns an env whose
    ``metadata`` property is immediately authoritative. POST /instances
    only commits an admission slot and runs the env's ``__init__`` — no
    physical resource (browser / docker / emulator) is allocated until
    the first ``reset()``, matching direct-mode laziness.
    """

    def __init__(
        self,
        server_url: str,
        env_key: str,
        *,
        metadata: LiteBaseMetadata | None = None,
        token: str | None = None,
        session_id: str | None = None,
        timeout: float = 600.0,
        **env_kwargs: Any,
    ) -> None:
        self._url = server_url.rstrip("/")
        self._env_key = env_key
        self._env_kwargs = env_kwargs
        self._session_id = session_id or os.environ.get("SESSION_ID", "default")
        self._metadata = metadata  # registry-time fallback, overwritten on create

        token = token if token is not None else os.environ.get("CUA_LITE_ENV_SERVER_TOKEN", "")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._timeout = timeout
        # Connection-pool sizing constants — see module top.
        self._client = httpx.AsyncClient(
            headers=self._headers,
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=_HTTPX_MAX_CONNECTIONS,
                max_keepalive_connections=_HTTPX_MAX_KEEPALIVE,
                keepalive_expiry=HTTPX_KEEPALIVE_EXPIRY_SEC,
            ),
        )
        self._id: str | None = None  # set by _eager_create_sync below

        # Eager create — fail-fast + mental-model parity with direct mode.
        # See class docstring. Uses a one-shot sync httpx.Client so this
        # works from any caller context (sync entry, or async caller that
        # wraps gym.make in asyncio.to_thread, as scripts/rollout.py
        # does). The async self._client above is reserved for reset/step/
        # close — it never participates in this synchronous boot path.
        self._eager_create_sync()

    def _eager_create_sync(self) -> None:
        """Synchronously POST /instances. Populates self._id and overwrites
        self._metadata with the server's authoritative response.

        Retries HTTP 503 (admission-queue full / pool exhausted) with
        a jittered backoff bounded by :data:`_EAGER_CREATE_DEADLINE_S`
        (10 min default). On deadline, raises ``CapacityExhausted`` so
        gym.make() fails loudly instead of hanging forever — a
        permanently-full server is an operator problem, not a per-task
        retry condition.

        Other statuses (4xx / non-503 5xx) raise immediately so the
        caller sees the failure at gym.make time instead of mysteriously
        at first reset().
        """
        deadline = time.monotonic() + _EAGER_CREATE_DEADLINE_S
        backoff = 0.5
        max_backoff = 5.0
        last_503: CapacityExhausted | None = None
        with httpx.Client(headers=self._headers, timeout=self._timeout) as sc:
            while True:
                try:
                    r = sc.post(
                        f"{self._url}/instances",
                        json={
                            "env_key": self._env_key,
                            "env_kwargs": self._env_kwargs,
                            "session_id": self._session_id,
                        },
                    )
                except httpx.ConnectError as e:
                    raise RuntimeError(
                        f"env-server unreachable at {self._url} ({e}).\n"
                        f"  Either it is NOT RUNNING, or STILL STARTING — the server "
                        f"registers every env (~1 min for large task sets, e.g. webgym) "
                        f"BEFORE it begins listening, so a freshly-(re)started server "
                        f"refuses connections until ready. Retry shortly if you just "
                        f"started it; else verify CUA_LITE_ENV_SERVER_URL + that it's up."
                    ) from e
                except httpx.TimeoutException as e:
                    raise RuntimeError(
                        f"env-server at {self._url} timed out creating an instance ({e}); "
                        f"it may be overloaded or still starting — retry shortly."
                    ) from e
                if r.status_code == 503:
                    # Prefer the server's Retry-After hint over our local
                    # backoff so synchronized re-storms walk apart at the
                    # server's pace rather than the client's.
                    retry_after = _parse_retry_after(
                        r.headers.get("Retry-After")
                    )
                    sleep_s = max(retry_after, backoff)
                    last_503 = _capacity_exhausted_from_response(r, retry_after)
                    # Jitter: ±50% on top of the chosen sleep, prevents
                    # all clients colliding on the boundary tick.
                    sleep_s = sleep_s * (0.5 + random.random())
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or sleep_s >= remaining:
                        logger.warning(
                            "eager-create exhausted %.0fs deadline waiting for "
                            "capacity; last 503=%s",
                            _EAGER_CREATE_DEADLINE_S, last_503.what,
                        )
                        raise last_503
                    logger.info("env-server 503 (%s); sleeping %.1fs",
                                last_503.what, sleep_s)
                    time.sleep(sleep_s)
                    backoff = min(backoff * 2.0, max_backoff)
                    continue
                if r.status_code >= 400:
                    # Log FIRST, above every branch below that raises — including
                    # the 501, whose body carries the EnvDepsMissingError
                    # what/install/see text that a client-only rollout process
                    # (no env package imported) cannot reconstruct from the
                    # status alone.
                    _log_error_response(r, "eager-create")
                    # Typed body FIRST: every create-time denial this server
                    # emits (EnvUnavailable 501 for blocked / not-allowed /
                    # deps-missing, MakeFailed 400, EnvAccessDenied 403 …)
                    # carries an ``error_type``, so it reconstructs as its real
                    # class with its real ``retryable``.
                    _raise_typed_error_if_any(r)
                    if r.status_code == 501:
                        # A typed body wins above; this arm is for version skew --
                        # an OLDER env-server, or a node body ``fleet.py`` forwards
                        # verbatim, can answer 501 with no ``error_type``. Without
                        # it that becomes an ``httpx.HTTPStatusError``, which
                        # ``is_retryable`` answers True for, so the caller burns its
                        # whole attempt budget on a permanent condition.
                        # 501 = env not served here. NOT 404: on reset/step that is
                        # instance-gone, which IS recoverable and must fall through.
                        raise _env_unavailable_from_response(r)
                r.raise_for_status()
                data = _response_json(r, "create")
                self._id = data["id"]
                if data.get("metadata"):
                    # Server response is authoritative — env_kwargs may have shaped it
                    # (e.g. extra_tools, valid_actions resolved per yaml config).
                    self._metadata = metadata_from_dict(data["metadata"])
                logger.info(
                    "LiteEnvClient created id=%s env_key=%s session_id=%s",
                    self._id, self._env_key, self._session_id,
                )
                return

    @property
    def metadata(self) -> LiteBaseMetadata:
        """Runtime metadata returned by the server when the instance was created."""
        assert self._metadata is not None  # populated by eager-create
        return self._metadata

    @property
    def metadata_or_none(self) -> LiteBaseMetadata | None:
        """Nullable, non-asserting sibling of :attr:`metadata`: the
        LIVE metadata from the ``POST /instances`` create response when the
        eager-create populated it, else the registry-time stub — which is
        ``None`` under ids-only enumeration. Public so the factory (and any
        other caller) never reaches into the private backing field."""
        return self._metadata

    def _runtime_metadata(self) -> LiteBaseMetadata:
        # Satisfies the base ABC; unreachable via the base ``metadata``
        # property (this class's own server-computed property shadows it).
        return self.metadata

    async def _post_with_retry(
        self,
        path: str,
        json_body: dict[str, Any] | None,
        op_name: str,
        *,
        replay_safe: bool,
    ) -> httpx.Response:
        """Shared retry loop for ``/reset`` and ``/step``.

        Every retry here re-sends the SAME POST, so both retry classes below turn
        on one property: ``replay_safe`` — may this request be sent again when we
        cannot prove the first one didn't take effect? ``/reset`` is replay-safe (a
        second reset re-establishes the same initial state); ``/step`` is not
        (without server-side request-id dedupe a replay can apply the same env
        action twice while the trajectory records one action).

        The two classes differ in how they're BOUNDED:

        * **HTTP 503** (server-typed capacity exhaustion): honor
          ``Retry-After`` exactly; loop until either a non-503 lands or the
          wall-clock deadline ``_CLIENT_503_DEADLINE_S`` would be exceeded by
          the next hinted sleep. On deadline: raise
          :class:`CapacityExhausted` (typed for the training loop's
          group-redraw filter).
        * **Transport drop** (:func:`~lite.gym.utils.backend.rpc.may_reissue`):
          exp-backoff per ``_client_retry_delay_s``, bounded by attempt count
          (``_CLIENT_RETRY_ATTEMPTS``) since these don't carry a server
          hint to follow. This hop's transport is httpx, so in practice that is
          the ``NetworkError`` family, ``RemoteProtocolError``, and
          ``ConnectTimeout`` (the one timeout that proves no request was sent).

        * **Client-side read timeout** (``httpx.ReadTimeout``): NOT a transport
          failure and NOT retried, regardless of ``replay_safe`` — the server is
          likely still executing the call and still holding its ``reset_sema``
          slot, so a re-POST stacks work on an already-slow server. Converted to a typed
          :class:`~lite.gym.remote.errors.RemoteCallTimeout`, which documents
          the client-vs-server budget asymmetry and tells the caller to
          ``close()`` before re-making.

        Other 4xx/5xx surface immediately — they're not retryable.
        """
        attempts_503 = 0
        attempts_transient = 0
        deadline = time.monotonic() + _CLIENT_503_DEADLINE_S
        while True:
            try:
                r = await self._client.post(f"{self._url}{path}", json=json_body)
                if r.status_code == 503:
                    attempts_503 += 1
                    retry_after = _parse_retry_after(r.headers.get("Retry-After"))
                    exc = _capacity_exhausted_from_response(r, retry_after)
                    if not replay_safe:
                        logger.warning(
                            "%s 503 on inst %s; not retrying non-idempotent op: %s",
                            op_name, self._id, exc.what,
                        )
                        raise exc
                    remaining = deadline - time.monotonic()
                    # If sleeping the hinted Retry-After would push past
                    # the deadline, there's no point waiting — the next
                    # request would land after our caller has given up.
                    # Strict ``>``: ``retry_after == remaining`` still
                    # lands at the deadline boundary which is legal
                    # (the request itself takes >0 ms, but the caller
                    # decides whether to honor that — server's hint
                    # said this is the right pause).
                    if retry_after > remaining:
                        logger.warning(
                            "%s 503 deadline reached on inst %s after %d attempts "
                            "(server hint %.1fs > %.1fs remaining): %s",
                            op_name, self._id, attempts_503,
                            retry_after, max(remaining, 0.0), exc.what,
                        )
                        raise exc
                    logger.info(
                        "%s 503 on inst %s; retry %d after %.1fs (server hint, "
                        "%.0fs deadline remaining)",
                        op_name, self._id, attempts_503,
                        retry_after, remaining,
                    )
                    await asyncio.sleep(retry_after)
                    continue
                # Generic typed-error propagation: any server-raised LiteGymError
                # (other than the 503/CapacityExhausted handled above) arrives as
                # a >=400 body tagged with its class name — including typed
                # typed statuses (404 InstanceGone: retryable, re-make recovers;
                # 409 ProtocolMisuse: terminal). Reconstruct it typed and raise.
                # An unknown ``error_type`` on a 500 is the envelope for a
                # non-registered server exception (e.g. a bare RuntimeError in
                # env.step()) — wrap it as RemoteEnvError so the caller sees
                # the original class name, not an opaque HTTPStatusError.
                if r.status_code >= 400:
                    _log_error_response(r, op_name)
                    _raise_typed_error_if_any(r)
                # Anything left (non-JSON body, proxy errors, …) surfaces via
                # raise_for_status; the 404/501 → terminal EnvUnavailable
                # mapping lives only on create + the catalog path, where those
                # statuses genuinely mean "env not served here".
                r.raise_for_status()
                return r
            except LiteGymError:
                # Already-typed errors from the branches above (CapacityExhausted
                # via the 503 path, any other LiteGymError via the generic path) —
                # re-raise so the transient-retry ``except`` below never swallows them.
                raise
            except httpx.ReadTimeout as e:
                # Request delivered, no reply inside OUR deadline — the server is
                # very likely still executing it (its /reset budget is ~4x ours).
                # Type it so the caller stops guessing: see RemoteCallTimeout.
                # ConnectTimeout / PoolTimeout / WriteTimeout are NOT caught here.
                # ConnectTimeout established no connection at all, so it is a
                # re-issuable transport failure (``is_transport_error`` +
                # ``_NEVER_REACHED``) and falls through to the retry below; the
                # other two fall through as non-transport and surface as-is.
                logger.warning(
                    "%s timed out client-side on inst %s after %.0fs; the server "
                    "may still be running it — close() before re-making",
                    op_name, self._id, self._timeout,
                )
                raise _remote_errors.RemoteCallTimeout(
                    f"{op_name}: no response within the client timeout "
                    f"({self._timeout:.0f}s) — the server may still be executing "
                    f"this call ({type(e).__name__})",
                    op_name=op_name,
                ) from e
            except Exception as e:
                if not may_reissue(e, replay_safe=replay_safe):
                    raise
                attempts_transient += 1
                if attempts_transient >= _CLIENT_RETRY_ATTEMPTS:
                    raise
                delay = _client_retry_delay_s(attempts_transient - 1)
                logger.warning(
                    "%s transient %s on inst %s; retry %d/%d after %.1fs",
                    op_name, type(e).__name__, self._id,
                    attempts_transient, _CLIENT_RETRY_ATTEMPTS, delay,
                )
                await asyncio.sleep(delay)

    async def _post_frame_with_retry(
        self,
        path: str,
        json_body: dict[str, Any] | None,
        op_name: str,
        decode,
        *,
        replay_safe: bool,
    ) -> LiteEnvObservation | LiteEnvStepResult:
        """POST reset/step and decode its binary frame.

        `_post_with_retry` handles HTTP/network transients before returning a
        200 response. A third live failure mode happens after that: the server
        accepted the operation but the 200 frame was cut short on the wire.
        The decode helper maps only that case to RemoteEnvError; JSON-200 and
        bad-magic/version stay terminal RemoteWireProtocolError.

        This layer takes the SAME ``replay_safe`` it forwards: a truncated 200 is
        the case where the operation provably DID execute, so re-sending it is no
        safer than re-sending after a network drop.
        """
        attempts_frame = 0
        while True:
            r = await self._post_with_retry(
                path,
                json_body,
                op_name,
                replay_safe=replay_safe,
            )
            try:
                return decode(r, op_name)
            except RemoteEnvError as exc:
                if exc.original_type != "MalformedServerResponse" or not replay_safe:
                    raise
                attempts_frame += 1
                if attempts_frame >= _CLIENT_RETRY_ATTEMPTS:
                    raise
                delay = _client_retry_delay_s(attempts_frame - 1)
                logger.warning(
                    "%s malformed 200 frame on inst %s; retry %d/%d after %.1fs: %s",
                    op_name, self._id, attempts_frame, _CLIENT_RETRY_ATTEMPTS,
                    delay, exc.what,
                )
                await asyncio.sleep(delay)

    async def reset(self) -> LiteEnvObservation:
        """Reset the remote instance and return the decoded initial observation."""
        # Replay-safe: /reset is idempotent — a second reset re-establishes the
        # same initial state, so a retry can at worst cost time.
        return await self._post_frame_with_retry(
            f"/instances/{self._id}/reset",
            json_body=None,
            op_name="/reset",
            decode=_decode_reset_frame,
            replay_safe=True,
        )  # type: ignore[return-value]

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        """Send one non-replay-safe action batch to the remote instance."""
        if self._id is None:
            raise RuntimeError("step() called before reset()")
        actions_list = list(actions)
        _raise_if_step_body_not_json_serializable(actions_list)
        # NOT replay-safe: without request-id dedupe, re-sending can apply the
        # same env action twice while the trajectory records one action. Holds for
        # every failure mode -- a 503 is not proof of pre-execution decline
        # (env-internal CapacityExhausted can escape after partial side effects),
        # and a truncated 200 frame is proof the step already ran.
        return await self._post_frame_with_retry(
            f"/instances/{self._id}/step",
            json_body={"actions": actions_list},
            op_name="/step",
            decode=_decode_step_frame,
            replay_safe=False,
        )  # type: ignore[return-value]

    async def close(self) -> None:
        """Close the remote instance, retaining its identity if DELETE fails."""
        # DELETE uses a fresh client so close still works after a cancelled
        # in-flight reset/step leaves the long-lived httpx client unusable.
        # ``asyncio.shield`` protects the release call from caller cancellation.
        try:
            if self._id is not None:
                token = self._client.headers.get("Authorization")
                headers = {"Authorization": token} if token else {}
                async with httpx.AsyncClient(headers=headers, timeout=30.0) as fresh:
                    response = await asyncio.shield(
                        fresh.delete(f"{self._url}/instances/{self._id}")
                    )
                    if response.status_code >= 400:
                        _log_error_response(response, "close")
                        _raise_typed_error_if_any(response)
                    response.raise_for_status()
                # Only an acknowledged DELETE releases the retry identity.
                self._id = None
        finally:
            # Close the local pool even when DELETE fails. A subsequent close
            # uses a fresh client with the retained identity and auth headers.
            # Bound teardown because cancelled connections can stall aclose.
            try:
                await asyncio.wait_for(self._client.aclose(), timeout=5.0)
            except Exception:
                pass


def register_from_server(env_id: str, server_url: str) -> None:
    """Fetch task list from the env server and register stub specs locally.

    Called by :func:`lite.gym.registry._import_env` whenever
    ``CUA_LITE_ENV_SERVER_URL`` is set — i.e. on every task-enumeration
    in remote-client mode, NOT only as a fallback after a failed local
    import. Lets the slime training container skip env-specific Python
    extras entirely while still being able to enumerate tasks for rollouts.

    The stub ``entry_point`` is never actually called in normal use:
    ``gym.make()`` short-circuits to :class:`LiteEnvClient` whenever
    ``CUA_LITE_ENV_SERVER_URL`` (or kwarg ``env_server_url``) is set. The
    placeholder only fires if someone calls ``make()`` after unsetting the
    env var — in which case a clear error beats a cryptic ``TypeError``.
    """
    # Late import to break the registry ↔ remote cycle.
    from lite.gym.registry import register, registry

    token = os.environ.get("CUA_LITE_ENV_SERVER_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"{server_url.rstrip('/')}/envs/{env_id}/tasks"
    # Wrap the HTTP call so connection / auth / not-found failures surface
    # with actionable guidance instead of a raw httpx traceback. The
    # refactor that made remote-mode the default routing path (rather than
    # a "deps missing" fallback) means most users will hit this code, so
    # the error messages have to point at the right config knob.
    try:
        resp = httpx.get(url, headers=headers, timeout=30.0)
    except httpx.ConnectError as e:
        raise EnvServerUnavailable(
            f"env-server unreachable at {server_url} ({e}).\n"
            f"  Either it is NOT RUNNING, or it is STILL STARTING — the server "
            f"registers every env (which can take ~1 min for large task sets, e.g. "
            f"webgym's ~293k) BEFORE it begins listening, so a freshly-(re)started "
            f"server refuses connections until ready. Check: (1) curl "
            f"{server_url.rstrip('/')}/host_status — retry shortly if you just "
            f"started it, (2) CUA_LITE_ENV_SERVER_URL host:port, (3) no firewall."
        ) from e
    except httpx.TimeoutException as e:
        raise EnvServerUnavailable(
            f"env-server at {server_url} timed out after 30s ({e}).\n"
            f"  It may be overloaded, stuck, or STILL STARTING — a server registering "
            f"a large task set (e.g. webgym's ~293k) before it begins serving can "
            f"exceed 30s. Retry shortly if you just (re)started it; otherwise curl "
            f"{server_url.rstrip('/')}/host_status to confirm it's responsive."
        ) from e
    except httpx.TransportError as e:
        raise EnvServerUnavailable(
            f"env-server at {server_url} is not reachable ({type(e).__name__}: {e}).\n"
            f"  Check CUA_LITE_ENV_SERVER_URL — it must include the scheme "
            f"(http://host:port) — and that no proxy sits in front of it."
        ) from e

    if resp.status_code == 401:
        token_set = "set" if token else "unset"
        raise EnvServerUnavailable(
            f"env-server at {server_url} returned 401 Unauthorized.\n"
            f"  CUA_LITE_ENV_SERVER_TOKEN is {token_set}. "
            f"Check it matches the server's expected token (operator config)."
        )
    if resp.status_code in (404, 501):
        raise EnvUnavailable(
            f"env-server at {server_url} doesn't serve env_id={env_id!r} "
            f"(HTTP {resp.status_code}) — NO SERVICE (terminal; retry won't help). "
            f"The env isn't registered/installed on this server. Check: (1) env_id "
            f"spelling, (2) the server has this env installed "
            f"(curl {server_url.rstrip('/')}/envs | jq 'keys') — a freshly "
            f"(re)started server only registers envs that were installed at its "
            f"launch, so install-then-restart if you just added one, (3) URL "
            f"points at the intended server.",
            status_code=resp.status_code,
        )
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise EnvServerUnavailable(
            f"env-server at {server_url} returned HTTP {resp.status_code} for "
            f"env_id={env_id!r}: {e}.\n"
            f"  Body excerpt: {resp.text[:200]!r}"
        ) from e
    try:
        data = resp.json()
    except ValueError as e:
        raise EnvServerUnavailable(
            f"env-server at {server_url} returned a non-JSON body for env_id={env_id!r} "
            f"({e}) — most often a proxy or error page rather than the server itself."
        ) from e

    if "splits" not in data:
        # Fail LOUD on version skew instead of a bare KeyError. 'splits'
        # is the one REQUIRED /tasks key; the optional keys below (metadata /
        # kwargs / env_make_kwargs / env_supported_kwargs) all have documented
        # empty-fallbacks, so an OLDER server that omits any of them degrades
        # instead of failing.
        raise EnvServerUnavailable(
            f"env-server at {server_url} returned a /tasks payload without the "
            f"required 'splits' key (got keys: {sorted(data)}) — server/client "
            f"version mismatch. Upgrade the server (or match the client version) "
            f"so both speak the same /tasks schema."
        )
    splits: dict[str, list[str]] = data["splits"]
    metadata_blobs: dict[str, dict[str, Any]] = data.get("metadata") or {}
    # Per-task carried make kwargs (wrapper-owned config such as timeouts, plus
    # env-owned make config such as cursor) the server carries so the stub spec
    # resolves the SAME config as a direct make().
    # ``.get(... ) or {}`` keeps back-compat with an older server that omits it.
    kwargs_blobs: dict[str, dict[str, Any]] = data.get("kwargs") or {}
    # env-wide make-kwarg defaults (layer 2) — apply so the client's make()
    # resolves them just like direct mode (the stub never imports the env module).
    registry.set_env_make_kwargs(env_id, data.get("env_make_kwargs") or {})
    # env-owned soft-kwarg capability (layer 3). ``.get(...) or []`` keeps the same
    # skew back-compat as the sibling keys above: an OLDER env-server omits it and
    # every env then advertises no soft kwargs, so a per-group ``seed`` is skipped
    # rather than rejected. The registry still unions in whatever the locally
    # registered specs declare, so the empty default narrows, never falsifies.
    registry.set_env_supported_kwargs(env_id, data.get("env_supported_kwargs") or [])
    for split, task_ids in splits.items():
        for tid in task_ids:
            md = (
                metadata_from_dict(metadata_blobs[tid])
                if tid in metadata_blobs else None
            )
            register(
                f"{env_id}@{tid}",
                entry_point=_remote_only_placeholder,
                split=split,
                metadata=md,
                **kwargs_blobs.get(tid, {}),
            )


def _remote_only_placeholder(**_kwargs: Any) -> LiteBaseEnv:
    raise RuntimeError(
        "Local env instantiation not available — env deps not installed. "
        "Set CUA_LITE_ENV_SERVER_URL to instantiate remotely via gym.make()."
    )
