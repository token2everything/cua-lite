"""Shared error taxonomy for lite.gym — the ONE contract for how an env or the
env-server signals failure, and how the remote client + training loop must react.

Four categories, organized by the caller's correct REACTION (not by where they
are raised). When you add a failure path, pick the category by answering "what
should the caller DO?" and raise the matching type — NEVER a bare ``RuntimeError``
/ ``ConnectionError`` on a served surface, because the server maps an untyped
exception to a TERMINAL HTTP 500, silently breaking the retry contract.

1. TERMINAL — NO SERVICE (do NOT retry; an operator must install/enable the env)
   * :class:`EnvDepsMissingError` — the env's Python deps / docker image / data
     aren't installed. Raised PRE-LAUNCH (import / ``ensure``). ``ImportError``
     subclass so the server's create path catches it in one ``except ImportError``.
     It has NO wire identity of its own: every served surface converts it with
     :meth:`EnvDepsMissingError.as_env_unavailable` so it crosses the RPC as
     :class:`EnvUnavailable` (HTTP 501) — see that class's docstring for why.
   * :class:`EnvUnavailable` — the env_id is not served by THIS deployment (not
     registered, hard-blocked to direct-mode, not on the ``--env-ids``
     allow-list, or deps-missing). The client re-raises it typed from a terminal
     server status: 501 on CREATE, and 404/501 on the CATALOG
     (task-enumeration) path (404 there = not registered; CREATE never emits
     404). Retrying never helps.
   * :class:`EnvAccessDenied` — the caller's bearer token does not own this
     instance. A permanent misconfiguration, not a per-request
     failure; retrying with the same token fails identically.

2. RECOVERABLE — TRANSIENT (retry safe operations, or redraw the rollout)
   * :class:`CapacityExhausted` — a bounded resource is *momentarily* unavailable:
     a pool / port / throttle is full, OR a just-started backend (container,
     emulator, web stack) is still WARMING UP / not-yet-HTTP-ready. Carries
     ``retry_after_s``. Server → HTTP 503 + Retry-After; the client retries
     retry-safe endpoints such as ``reset`` to its deadline, then re-raises it
     typed so the training loop can group-redraw. Non-idempotent ``step`` 503s
     surface typed immediately until server-side request-id dedupe exists.
     **Every docker / VM / emulator env MUST signal boot/warming with this type**
     (use :meth:`CapacityExhausted.warming`) — a plain error there becomes a
     terminal 500 (see e.g. webgym's prior cold-start bug).

3. RECOVERABLE — INSTANCE GONE (re-create the instance)
   * :class:`InstanceGone` — the server returns a typed 404 from
     ``_require_own_env`` when the instance was lost (server restart / idle-TTL
     reap / DELETE). ``retryable = True``: the client's re-make path recovers.
     It must NOT be mapped to the terminal :class:`EnvUnavailable`.

4. VOID EPISODE — UNSCOREABLE (drop the trajectory; do NOT retry)
   * :class:`EnvBlocked` — an env-side condition outside the agent's control made the
     task unscoreable (e.g. WebGym's live site bot-blocked the agent). Excluded from the
     score like any error, and ``retryable = False`` (a re-run re-blocks identically, so
     :func:`is_retryable` skips it).
   * ``Cua*TaskError`` / ``CuaWorldVerifierError`` — deterministic setup,
     reward, or verifier failures where no trustworthy score was computed.
     Also non-retryable, so rollout drops the sample instead of
     recording a false reward-0.

A category is signalled by the TYPE, never by the HTTP status. Several
subclasses set presentation statuses; only :class:`CapacityExhausted`,
:class:`EnvUnavailable` and :class:`InstanceGone` are consumed by status number
without reading the typed body, and :class:`LiteGymError` says why.

Outside the four categories, :class:`EnvServerUnavailable` is the one typed
failure that is about the DEPLOYMENT rather than an env: the whole env-server is
unreachable / refuses the token / is version-skewed. It is client-local (it never
crosses the wire) and so carries no ``error_type``.

Stateless graders (``screenspot_pro`` / ``osworld_g``) raise NONE of these — no
bounded resource and nothing to warm.

Message paths differ by audience. **Direct mode** (``gym.make`` in-process): the
caller is the developer/installer — show ``what`` + the ``install`` command + the
README path. **Server mode** (``POST /instances``): the caller is a remote client
who can't install on someone else's server — show only that the env is unsupported
here + ``what``; the operator consults the env's README out-of-band.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Literal

from lite.core.errors import LiteContractError, ToolCallValidationError

# ---------------------------------------------------------------------------
# Generic typed-error propagation across the env-server RPC boundary.
#
# A subclass of :class:`LiteGymError` is serialized by ONE generic server
# handler to ``http_status`` + a JSON body tagged with the class name, and the
# remote client reconstructs it typed from that body — so a server-raised typed
# error can never silently become a terminal HTTP 500 just because nobody wired
# a per-type handler. Errors needing bespoke wire behavior (e.g.
# ``CapacityExhausted``'s Retry-After + /metrics attribution) keep a
# more-specific handler, which FastAPI prefers over the generic one.
# ---------------------------------------------------------------------------

_LITE_ERROR_REGISTRY: dict[str, type[LiteGymError]] = {}


class FailureCategory(StrEnum):
    """Shared model-vs-infra action-boundary categories."""

    MODEL_ACTION_ERROR = "model_action_error"
    MODEL_OUTPUT_ERROR = "model_output_error"
    INFRA_FAILURE = "infra_failure"


class PairableModelActionError(ValueError):
    """A model-caused action error with a valid call_id for role:tool feedback."""

    failure_category = FailureCategory.MODEL_ACTION_ERROR
    pairable = True
    retryable = False


class TrueInfraFailure(RuntimeError):
    """A real backend/env failure that should raise through infra policy."""

    failure_category = FailureCategory.INFRA_FAILURE
    pairable = False
    retryable = True


def failure_category(exc: BaseException) -> FailureCategory | None:
    """Return a shared action-boundary category carried by an exception type.

    Two sources, both keyed on the TYPE. Gym's own errors declare the category as
    a class attribute; a core contract error is classified HERE, for the same
    reason :func:`is_retryable` classifies it here — the category name is gym
    vocabulary and ``lite.core`` cannot import this module (see
    ``lite/core/errors.py``).

    There is deliberately no bare-string arm. Every declaration in the repo is a
    :class:`FailureCategory` member; the arm that used to resolve a string with
    ``FailureCategory(category)`` inside ``except ValueError: return None``
    existed for ONE hand-copied spelling in ``lite/core/errors.py``, and its
    effect was that a drifted copy classified as "no opinion" in silence.
    """
    category = getattr(exc, "failure_category", None)
    if isinstance(category, FailureCategory):
        return category
    if isinstance(exc, ToolCallValidationError):
        return FailureCategory.MODEL_OUTPUT_ERROR
    return None


def register_lite_error(cls):  # decorator
    _LITE_ERROR_REGISTRY[cls.__name__] = cls
    return cls


def lite_error_from_payload(payload: dict):
    """Reconstruct a typed :class:`LiteGymError` from a server error body.

    Returns the typed exception, or ``None`` when the payload carries no known
    ``error_type`` (e.g. a 404 instance-gone body) so the caller can fall
    through to its non-typed handling.
    """
    cls = _LITE_ERROR_REGISTRY.get(payload.get("error_type", ""))
    return cls.from_payload(payload) if cls is not None else None


def is_retryable(exc: BaseException) -> bool:
    """The ONE retry predicate shared by every trajectory-level retry loop.

    Reads the ``retryable`` attribute off the exception's TYPE (so it works for
    any class — :class:`LiteGymError` subclasses, :class:`EnvDepsMissingError`
    (an ``ImportError``), etc. — without inheritance constraints). The default is
    asymmetric on purpose:

    * a **typed** gym error is a known condition → terminal unless it explicitly
      sets ``retryable = True``; a forgotten flag fails safe (no wasted retries);
    * an **untyped/unknown** exception (a raw crash, a 404 instance-gone) → ``True``:
      a fresh re-run might recover, and the caller's attempt cap bounds the waste.

    A :class:`lite.core.errors.LiteContractError` is neither: it is a CLIENT
    FAULT — caller-supplied data violated a Lite protocol contract, so a re-run
    rebuilds the identical wrong shape and fails identically. It is classified
    HERE, not by an attribute on the core class, because
    ``lite.core.errors`` states that retry policy is gym's to decide (its
    ``failure_category`` slot defaults to ``None`` — "no opinion, gym classifies
    it") and core must not depend on this taxonomy. Reached in practice from
    ``BaseAgent._predict_with_details``, whose ``_render_and_prep`` calls
    ``referenced_images_in_message_order`` UNGUARDED, so an out-of-range image index
    escapes ``sample()`` into the trajectory loop's ``except Exception``.

    NOT covered: the same error raised SERVER-side crosses the RPC as an
    unregistered ``error_type`` and the client rebuilds it as
    :class:`RemoteEnvError`, which is ``retryable = True`` by construction.
    Closing that needs a wire-registration decision, not a second isinstance arm.
    """
    if isinstance(exc, LiteContractError):
        return False
    return getattr(exc, "retryable", True)


class LiteGymError(Exception):
    """Base for typed failures that must survive the env-server RPC boundary.

    Subclass + ``@register_lite_error``; override
    ``_payload_extra``/``from_payload`` if it carries extra fields. One generic
    server handler serializes any subclass -> its status + a body tagged with the
    class name; the client reconstructs it typed -> no per-type plumbing, so a
    server-raised typed error can never silently become HTTP 500. Errors needing
    bespoke wire behavior (CapacityExhausted's Retry-After/metrics) keep a
    more-specific handler, which FastAPI prefers.

    DO NOT make ``http_status`` the category — ``error_type`` is AUTHORITATIVE,
    the status is PRESENTATION, so a new failure earns a registered name, not a
    number. Only three statuses have consumers that branch on the number WITHOUT
    reading the body: :class:`CapacityExhausted` **503** (the client's
    Retry-After loop), :class:`EnvUnavailable` **501/404** (the client's
    untyped-501 arm and ``register_from_server``), :class:`InstanceGone` **404**
    (``fleet.py`` evicts its sticky routing entry on a node 404). The base
    default remains 500 for otherwise unclassified typed errors; the client wraps
    an unknown ``error_type`` as retryable on 500/502 only.

    ``retryable`` encodes the caller's RETRY reaction (the taxonomy's spine, made
    machine-readable): a typed gym error is TERMINAL by default — a re-run fails
    identically. Subclasses override ONLY to deviate (for example
    :class:`CapacityExhausted`, :class:`InstanceGone`, and
    :class:`EnvDesktopCrashed`). Terminal subclasses just inherit this default
    (:class:`EnvDepsMissingError` restates it solely because it is an
    ``ImportError``, not a ``LiteGymError``, so it can't inherit). Read it via
    :func:`is_retryable`, never by ``isinstance`` at a retry site.

    This is the abstract base, never raised on its own — only concrete
    subclasses carry a wire identity."""

    http_status: int = 500
    retryable: bool = False

    def __init__(self, what: str, *, message: str | None = None) -> None:
        self.what = what
        super().__init__(message if message is not None else what)

    def _payload_extra(self) -> dict:
        return {}

    def to_payload(self) -> dict:
        return {"error_type": type(self).__name__, "what": self.what, **self._payload_extra()}

    @classmethod
    def from_payload(cls, payload: dict):
        return cls(what=payload.get("what", ""))


@register_lite_error
class InstanceGone(LiteGymError):
    """The instance id is unknown to the server — restart, idle-TTL reap, or
    explicit DELETE. RECOVERABLE by re-making the env: ``retryable``
    is set explicitly so the trajectory loop's re-make path rests on a TYPED
    identity instead of the unknown→retry default a bare 404 relied on. Keeps
    its 404 (one of the base's three exceptions): ``fleet.py``'s sticky forward
    pops the routing entry on a node 404 without parsing the body."""

    http_status = 404
    retryable = True


@register_lite_error
class ProtocolMisuse(LiteGymError):
    """The client drove the instance API out of order — e.g. ``/step`` before
    ``/reset``. Terminal by the base default: retrying the same call
    fails identically; fix the caller. (The old naked 409 carried no
    ``error_type``, so ``is_retryable``'s unknown→retry default pointlessly
    retried it — the ``error_type`` is what fixed that, never the number.)"""

    http_status = 409


@register_lite_error
class MakeFailed(LiteGymError):
    """``gym.make`` raised on the server during ``POST /instances`` — bad
    env_kwargs, unknown task, or an env bug. Terminal for this create
    (a retry with identical inputs fails identically)."""

    http_status = 400


class RemoteEnvError(LiteGymError):
    """CLIENT-side wrapper for a server-raised exception that is NOT a
    registered :class:`LiteGymError`: the server serializes
    every unhandled exception as ``{error_type, what}`` and the client wraps
    unknown types in this, so a bare ``RuntimeError`` inside ``env.step()``
    reaches the caller with its original class name instead of as an opaque
    ``httpx.HTTPStatusError(500)``.

    Deliberately not ``@register_lite_error`` — it never originates server-side,
    so no server body is ever tagged with this name.
    ``retryable = True`` because an unregistered exception type is an unknown
    condition, and unknown → retry."""
    retryable = True

    def __init__(self, what: str, *, original_type: str) -> None:
        super().__init__(what, message=f"{original_type}: {what}")
        self.original_type = original_type


class RemoteWireProtocolError(LiteGymError):
    """CLIENT-side terminal protocol/version skew.

    Raised when the server answered successfully at HTTP level but the body is
    not the binary frame this client version speaks. This is not no-service
    (``EnvUnavailable``), and retrying the same client/server pair will not fix
    it; upgrade one side.

    Like :class:`RemoteEnvError` it is raised only CLIENT-side, about a response
    the server considered successful — the server never serializes a body tagged
    with this name.
    """

    def __init__(self, what: str) -> None:
        super().__init__(what)


# These CUA-Lite task/verifier failures originate in env adapters, but their
# wire identities must be registered in this always-imported module. Remote
# rollout clients do not import env packages when they fetch server-side task
# stubs, so env-local registration alone would degrade 422s to opaque
# ``HTTPStatusError`` and make deterministic setup/verifier failures retryable.
@register_lite_error
class CuaGymTaskError(LiteGymError):
    """A deterministic CUA-Gym setup/reward failure, not an agent failure."""

    http_status = 422

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        kind: str,
        returncode: int | None = None,
    ) -> None:
        self.phase = phase
        self.kind = kind
        self.returncode = returncode
        super().__init__(message)

    def _payload_extra(self) -> dict:
        return {
            "phase": self.phase,
            "kind": self.kind,
            "returncode": self.returncode,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> CuaGymTaskError:
        return cls(
            payload.get("what", ""),
            phase=payload.get("phase", "unknown"),
            kind=payload.get("kind", "unknown"),
            returncode=payload.get("returncode"),
        )


@register_lite_error
class ScaleCuaTaskError(LiteGymError):
    """A deterministic ScaleCUA task/setup/eval failure, not infrastructure."""

    http_status = 422

    def __init__(
        self,
        message: str,
        *,
        phase: str = "setup",
        kind: str = "task",
        returncode: int | None = None,
    ) -> None:
        self.phase = phase
        self.kind = kind
        self.returncode = returncode
        super().__init__(message)

    def _payload_extra(self) -> dict:
        return {
            "phase": self.phase,
            "kind": self.kind,
            "returncode": self.returncode,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> ScaleCuaTaskError:
        return cls(
            payload.get("what", ""),
            phase=payload.get("phase", "unknown"),
            kind=payload.get("kind", "unknown"),
            returncode=payload.get("returncode"),
        )


type CuaWorldVerifierPhase = Literal[
    "setup",
    "screenshot",
    "post_task",
    "verify",
]
type CuaWorldVerifierKind = Literal[
    "timeout",
    "spawn",
    "command_failed",
    "screenshot_failed",
    "verifier_raised",
    "bad_result",
]


@register_lite_error
class CuaWorldVerifierError(LiteGymError):
    """The CUA-World verifier could not produce a score."""

    http_status = 422

    def __init__(
        self,
        message: str,
        *,
        phase: CuaWorldVerifierPhase,
        kind: CuaWorldVerifierKind,
        task: str | None = None,
    ) -> None:
        self.phase: CuaWorldVerifierPhase | str = phase
        self.kind: CuaWorldVerifierKind | str = kind
        self.task = task
        super().__init__(message)

    def _payload_extra(self) -> dict:
        return {"phase": self.phase, "kind": self.kind, "task": self.task}

    @classmethod
    def from_payload(cls, payload: dict) -> CuaWorldVerifierError:
        return cls(
            payload.get("what", ""),
            phase=payload.get("phase", "unknown"),
            kind=payload.get("kind", "unknown"),
            task=payload.get("task"),
        )


class EnvDepsMissingError(ImportError):
    """Env discoverable but not launchable due to missing dependencies.

    Covers any pre-launch shortage:
      * Python package (e.g. ``desktop_env`` for OSWorld)
      * Docker image not built (e.g. ``cua-lite/androidlab:latest``)
      * Data files not downloaded (e.g. OSWorld-G ``data/`` tree)
      * External service unreachable

    Direct mode usage at the top of an env's ``main.py``::

        try:
            import desktop_env  # noqa: F401
        except ImportError:
            from lite.gym.errors import EnvDepsMissingError
            raise EnvDepsMissingError(
                what="OSWorld (desktop_env) package not installed",
                install="uv run --no-sync bash lite/gym/envs/osworld/scripts/install.sh",
                see="lite/gym/envs/osworld/README.md",
            )

    It must stay an ``ImportError`` so ``gym.make``'s callers can keep catching it
    as one, and its 3-arg ``__init__(what, install, see)`` carries two fields that
    :meth:`server_message` deliberately withholds from a remote caller.

    Its WIRE identity is :class:`EnvUnavailable` instead — the taxonomy already
    owns "terminal NO-SERVICE, HTTP 501, ``retryable = False``" and is
    registered with a ``from_payload``. Every server surface converts via
    :meth:`as_env_unavailable`, so the direct-mode message (with install/see)
    and the server-mode message (without) stay on their own sides of the RPC
    and the client gets a typed, non-retryable error instead of a
    ``RemoteEnvError`` that is retryable by construction.

    Args:
        what:    One-line description of what's missing (used in both
                 direct and server messages).
        install: Exact shell command to install — DIRECT MODE ONLY.
        see:     Repo-root absolute path to the env's README.md —
                 DIRECT MODE ONLY.
    """

    # Not a LiteGymError (it's an ImportError) — declare the attr so is_retryable
    # reads it by attribute; deps won't appear by retrying, so terminal.
    retryable: bool = False

    def __init__(self, what: str, install: str, see: str) -> None:
        self.what = what
        self.install = install
        self.see = see
        super().__init__(
            f"{what}\n"
            f"  Install with: {install}\n"
            f"  See:          {see}"
        )

    def server_message(self, env_id: str) -> str:
        """Client-facing message for the env-server's 501 response.

        Withholds ``install`` / ``see`` — the client is remote and
        can't install on the server. Operator consults the env's
        README out-of-band.
        """
        return (
            f"env_id={env_id!r} is not available on this server "
            f"({self.what}). Contact the server operator."
        )

    def as_env_unavailable(self, env_id: str) -> EnvUnavailable:
        """The wire form of this error: a registered, terminal 501.

        The conversion point for the served surfaces that need it — ``POST
        /instances``, ``/reset`` and ``GET /envs/{id}/tasks``, but NOT ``/step``.
        :meth:`server_message` keeps ``install``/``see`` off the wire, and a
        registered :class:`LiteGymError` serializes as ``{error_type:
        "EnvUnavailable", …}`` so the client reconstructs it with
        ``retryable = False``.
        """
        return EnvUnavailable(what=self.server_message(env_id), status_code=501)


@register_lite_error
class CapacityExhausted(LiteGymError):
    """An env's bounded internal resource was exhausted within the wait deadline.

    Raised by:

    * Pool-based envs (mobilegym, webgym) when their internal slot pool
      stays full past ``timeout_s``.
    * Cost-budget throttles (boot, active_op) when the configured budget
      stays full past the acquire deadline.
    * Container-backed envs (androidworld) on port-range exhaustion.
    * **Any docker / VM / emulator env whose just-started backend is still
      WARMING UP / not-yet-HTTP-ready during reset** — use the
      :meth:`warming` constructor. This is the universal boot/warming signal;
      a plain ``RuntimeError`` here would map to a terminal 500 instead.

    Env transport shape: the env-server's
    ``@app.exception_handler(CapacityExhausted)`` maps this to::

        HTTP/1.1 503 Service Unavailable
        Retry-After: <int(retry_after_s)>
        {
            "error_type": "CapacityExhausted",
            "detail": "<str(exc)>",
            "what": "<machine-actionable reason>",
            "retry_after_s": <float>,
            "layer": "<capacity layer>",
        }

    The client (:class:`lite.gym.remote.client.LiteEnvClient`) parses
    ``Retry-After`` and retries retry-safe endpoints such as ``reset`` until
    the configured wall-clock 503 deadline; on exhaustion it re-raises this
    exception (not generic ``HTTPStatusError``) so the training loop's
    group-redraw filter can match by type. Non-idempotent ``step`` requests
    surface typed 503s immediately until server-side request-id dedupe exists.

    Envs without a bounded internal resource (stateless graders like
    ``screenspot_pro`` / ``osworld_g``) do NOT raise this — L1 (host
    sensors) + L2 (in-flight cap) handle their backpressure.

    Args:
        what:          One-line description of the exhausted resource
                       (for logs and error messages).
        retry_after_s: Server's hint for how long the client should wait
                       before retrying. Should reflect the typical time
                       for a slot to free up (~60s for browser pools,
                       ~30s for port allocations).
        layer:         Which admission layer raised this 503. Used by
                       the env-server's ``/metrics`` to attribute 503s
                       (``emergency``/``capacity``/``docker_sema``/
                       ``env_internal``). Defaults to ``env_internal``
                       because env-side code (the most common raiser)
                       shouldn't have to know about the gate taxonomy.
                       Admission-gate raise sites set this explicitly.
    """

    http_status: int = 503
    retryable: bool = True   # the one transient: pool/port full, or backend warming

    def __init__(
        self, what: str, retry_after_s: float, layer: str = "env_internal",
    ) -> None:
        self.retry_after_s = float(retry_after_s)
        self.layer = layer
        super().__init__(
            what, message=f"{what} (retry_after_s={self.retry_after_s:.0f})",
        )

    def _payload_extra(self) -> dict:
        return {"retry_after_s": self.retry_after_s, "layer": self.layer}

    @classmethod
    def from_payload(cls, payload: dict) -> CapacityExhausted:
        return cls(
            what=payload.get("what", ""),
            retry_after_s=payload.get("retry_after_s", 0.0),
            layer=payload.get("layer", "env_internal"),
        )

    @classmethod
    def warming(cls, what: str, retry_after_s: float = 20.0) -> CapacityExhausted:
        """The canonical "backend still warming up / not-yet-HTTP-ready" transient.

        Sugar so EVERY docker / VM / emulator env's boot/ready path raises ONE
        consistent type during retry-safe startup paths such as ``reset`` —
        never a plain ``RuntimeError`` / ``ConnectionError`` (which the
        env-server maps to a terminal HTTP 500). The env-server still maps it
        to 503 + Retry-After exactly like any other ``CapacityExhausted``;
        ``layer`` stays the default ``env_internal`` (env-side code) so
        existing ``/metrics`` buckets are unchanged. If the same condition
        escapes from non-idempotent ``step``, the remote client surfaces it as
        a typed immediate failure rather than replaying the action.
        """
        return cls(what=what, retry_after_s=retry_after_s)


@register_lite_error
class EnvUnavailable(LiteGymError):
    """Terminal NO-SERVICE: the env_id is not served by this deployment.

    The crisp "no service" counterpart to :class:`CapacityExhausted`'s "still
    starting, retry". Raised CLIENT-SIDE when the env-server answers a terminal
    status for an env_id:

      * **404** — env_id not registered on this server (never installed there,
        or the server was started before the env's ``install.sh`` ran).
      * **501** — env hard-blocked to direct-mode, not on the server's
        ``--env-ids`` allow-list, or its deps / docker image / data aren't
        installed on the server (:meth:`EnvDepsMissingError.as_env_unavailable`).

    The 501 family is ONE status for all three: they are the same condition from
    the caller's side ("this deployment refuses this env_id, by configuration")
    and differ only in which operator knob caused it.

    It is also raised SERVER-side (unlike its client-only siblings) so those 501s
    carry an ``error_type`` the client reconstructs, rather than relying on the
    client special-casing the status number.

    Retrying does NOT help (unlike a 503): the operator must install/enable the
    env, or the caller must use a server that serves it. Typed — not a bare
    ``httpx.HTTPStatusError`` — so the rollout loop can tell "give up on this
    env" apart from ``CapacityExhausted``'s "wait, backend still warming up".

    Args:
        what:        One-line description (the server's ``detail`` when present).
        status_code: The terminal HTTP status (404 / 501) for log attribution.
    """

    def __init__(self, what: str, status_code: int) -> None:
        self.status_code = status_code
        self.http_status = status_code  # per-instance (404 vs 501) — overrides the class default
        super().__init__(what, message=f"{what} (HTTP {status_code})")

    def _payload_extra(self) -> dict:
        return {"status_code": self.status_code}

    @classmethod
    def from_payload(cls, payload: dict) -> EnvUnavailable:
        return cls(
            what=payload.get("what", ""),
            status_code=int(payload.get("status_code", payload.get("http_status", 501))),
        )


class EnvServerUnavailable(RuntimeError):
    """The WHOLE env-server is unusable — not one env_id being unserved.

    Raised CLIENT-side by ``register_from_server`` for a server that is
    unreachable, timed out, refused the token (401), answered a non-terminal
    error status, or speaks a skewed ``/tasks`` schema. Every one of those is a
    fact about the deployment, so it must not be reported as "this env is absent
    here" — that is :class:`EnvUnavailable`'s (404/501) claim, and it is the
    per-env one.

    It has NO wire identity (it never crosses the RPC — it *is* the failure to
    cross it) and stays a plain ``RuntimeError`` subclass so existing callers
    that only care that the fetch failed are unaffected. It exists so
    :func:`~lite.gym.registry._import_all` can tell a dead server (skip this env
    in the catalog probe, the same as today) from a BROKEN env module, which must
    propagate.
    """


@register_lite_error
class EnvAccessDenied(LiteGymError):
    """Terminal AUTHORIZATION failure on the env-server: this caller's bearer
    token may not touch this resource.

    Currently one condition: the instance id exists but belongs to ANOTHER
    token (``_require_own_env`` / ``DELETE``). It is a permanent
    misconfiguration — a wrong token, or a client holding an id it never
    created — so it is terminal by the base default.

    Typed for the same reason :class:`ProtocolMisuse` is: FastAPI's
    ``HTTPException`` body is ``{"detail": …}`` with no ``error_type``, so the
    client's typed reconstruction falls through to ``raise_for_status`` and
    ``is_retryable``'s unknown→True default re-runs the whole task against a token
    that will never be accepted.

    ``http_status = 403`` even though our own client reconstructs from
    ``error_type``: 4xx-vs-5xx is the client-fault/server-fault split, and
    answering a permission denial with 500 tells every proxy, monitor and curl
    user that the server is broken while it is working correctly.

    Args:
        what: One-line description (which id, and that it is owned elsewhere)
              — never echo the caller's token.
    """

    http_status = 403


@register_lite_error
class EnvBlocked(LiteGymError):
    """VOID EPISODE — an env-side condition outside the agent's control made the
    task unscoreable, so it must be DROPPED (not counted as a reward-0 model
    failure).

    It carries NO block-specific handling anywhere — nothing in the codebase
    branches on this type. It is simply an error that happens to be:
      * excluded from the score (like every error — ``error != None`` /
        empty-sample / not in ``n_trajs_valid``), and
      * **non-retryable** (``retryable = False``): re-running re-blocks
        identically, so the shared retry predicate :func:`is_retryable` skips it.
    Both behaviours follow from generic properties, not from its identity.

    It is typed (vs a bare exception) only so the generic :class:`LiteGymError`
    propagation carries it across the env-server RPC intact instead of degrading
    to an opaque 500.

    Current raiser: WebGym, when its episode-end judge finds the live site
    bot-blocked / access-denied the agent (``info.blocking.is_blocked``).
    """

    http_status = 422


@register_lite_error
class EnvDesktopCrashed(LiteGymError):
    """RETRYABLE env-infra failure: the container's GUI desktop session crashed
    after boot, so the booted observation is unusable.

    Concretely (lite.osworld): the GNOME-Shell session hit the "Oh no,
    something has gone wrong" fallback — typically host inotify exhaustion under
    concurrent boots (``fs.inotify.max_user_instances``). ``reset()`` would
    otherwise return a "successful" screenshot OF THE CRASH SCREEN, the agent
    flails on a dead desktop, and the task SILENTLY scores reward=0 — poisoning
    the dataset.

    Typed (not a bare ``RuntimeError`` → opaque HTTP 500) so the generic
    :class:`LiteGymError` propagation carries the message across the env-server
    RPC into the client's ``error.txt`` / logs intact. ``retryable = True`` (like
    :class:`CapacityExhausted`) so the rollout's attempt loop tears the container
    down and re-runs on a fresh one instead of writing a poisoned reward-0
    summary. Unlike ``CapacityExhausted`` it declares NO ``http_status``, so it
    keeps the base's default 500 rather than 503 — the client's Retry-After loop
    never sees it, and it raises immediately instead of burning same-instance
    retries on a desktop that is already dead. The typed body is what lets the
    client reconstruct it; an untyped 500 would degrade to a bare
    ``httpx.HTTPStatusError``."""

    retryable: bool = True


class ReconcileProbeError(Exception):
    """An env's ``live_ids`` probe could not enumerate its external world THIS cycle
    (e.g. ``docker ps`` / ``pgrep`` transiently failed). Raised by a DEDICATED env's
    ``live_ids`` instead of the old fail-closed ``None`` return, so ``None`` now has a
    single meaning ("no per-instance world", the base default). The reconcile loop
    (:func:`lite.gym.remote.reconcile.reconcile`) catches it and skips the cycle
    (``ReapReport(0)`` — byte-identical fail-closed: never a partial reap), so a docker
    hiccup never false-ghosts live instances.

    NOTE: a plain ``Exception``, NOT a :class:`LiteGymError` — it is an internal
    drift-reaper signal caught inside ``reconcile()``; it never crosses the env-server RPC
    nor enters the trajectory-error taxonomy (``retryable`` / score-exclusion), so it stays
    deliberately outside that hierarchy.
    """
