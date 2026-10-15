"""Remote-specific typed failures shared by the env-server and client."""

from __future__ import annotations

import math
from typing import Any

from lite.core.tools.calls import RUNTIME_RESULT_CALL_ID_KEY
from lite.gym.errors import FailureCategory, LiteGymError, register_lite_error

#: Bounds on :attr:`ModelOutputError.payload_metadata` so one malformed action
#: can never flood the error envelope (or the log line rendered from it).
PAYLOAD_METADATA_MAX_KEYS = 20
PAYLOAD_METADATA_MAX_VALUE_CHARS = 160


def payload_metadata_value(value: Any) -> Any:
    """Render one action field as a **transport-legal** ``payload_metadata`` value.

    ``payload_metadata`` is always about to cross a
    ``json.dumps(..., allow_nan=False)`` boundary — Starlette's ``JSONResponse``
    on the server hop, the client's own ``/step`` preflight on the other — so a
    non-finite float must never enter the blob in the first place. Coercing it
    here (at the single point of production) is why no consumer needs a repair
    step: a ``NaN`` that reached the server's 422 body would fail the response
    render itself, turning a typed 422 into a broken/empty reply.
    """
    if isinstance(value, str):
        return value[:PAYLOAD_METADATA_MAX_VALUE_CHARS]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return {"type": type(value).__name__}


def action_payload_metadata(index: int, action: Any) -> dict[str, Any]:
    """Bounded description of the ``/step`` action at ``index``, for
    :attr:`ModelOutputError.payload_metadata`.

    One implementation for both hops: the server describes the action it just
    rejected, the client describes the action its serialization preflight
    refused to send, and a debugger comparing the two envelopes must see the
    same shape.
    """
    metadata: dict[str, Any] = {"action_index": index}
    if not isinstance(action, dict):
        metadata["action_type"] = type(action).__name__
        return metadata

    keys = [str(key) for key in action.keys()]
    metadata["action_keys"] = sorted(keys)[:PAYLOAD_METADATA_MAX_KEYS]
    for key in ("id", "type", "call_id", "tool_call_id", RUNTIME_RESULT_CALL_ID_KEY):
        if key in action:
            metadata[key] = payload_metadata_value(action[key])

    function = action.get("function")
    if isinstance(function, dict) and "name" in function:
        metadata["name"] = payload_metadata_value(function["name"])
    return metadata


@register_lite_error
class ModelOutputError(LiteGymError):
    """Malformed model-emitted action payload at the remote step boundary."""

    http_status = 422
    retryable = False
    failure_category = FailureCategory.MODEL_OUTPUT_ERROR
    pairable = False

    def __init__(
        self,
        what: str,
        *,
        kind: str = "malformed_step_request",
        payload_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.kind = kind
        self.payload_metadata = payload_metadata or {}
        super().__init__(what)

    def _payload_extra(self) -> dict:
        out: dict[str, Any] = {"kind": self.kind}
        if self.payload_metadata:
            out["payload_metadata"] = self.payload_metadata
        return out

    @classmethod
    def from_payload(cls, payload: dict) -> ModelOutputError:
        return cls(
            payload.get("what", ""),
            kind=payload.get("kind", "malformed_step_request"),
            payload_metadata=payload.get("payload_metadata")
            if isinstance(payload.get("payload_metadata"), dict)
            else None,
        )


class RemoteCallTimeout(LiteGymError):
    """CLIENT-side: the request was delivered but no response arrived inside the
    client's own HTTP deadline (``httpx.ReadTimeout``).

    **This is not a server failure — it is the two sides' budgets disagreeing,
    and the asymmetry is deliberate.** The client's timeout is
    ``max(reset_timeout, step_timeout) + 60`` (``lite.gym.factory``; 660 s at
    the default ``reset_timeout=600``) — ONE reset plus slack. The server's
    ``/reset`` runs up to ``_OUTER_RETRY_ATTEMPTS`` (4) full resets *inside the
    held ``reset_sema`` slot*, so its worst case is ~4× the client's patience.
    Raising either number is a capacity decision, so this type names the gap
    rather than papering over it with a bigger constant: the operation is still
    running server-side, holding its slot, and the instance's state is unknown.

    Raised only client-side, about a response that never arrived; no server body
    is ever tagged with this name. A READ timeout is also deliberately NOT a
    transport failure — see :attr:`retryable` below.
    """

    #: TRAJECTORY-level, not transport-level -- the two answers differ and that is
    #: the whole point. The rollout loop may abandon this instance and make a
    #: fresh one (that recovers); the transport must NOT re-POST the call it just
    #: timed out on (that may already be executing), which is why the timeout
    #: families whose peer may still be working (read / write / pool, and the
    #: builtin) are excluded from ``is_transport_error`` and therefore from
    #: ``may_reissue`` on both hops. ``httpx.ConnectTimeout`` is NOT in that set
    #: and never reaches this class: no connection, so no request and no
    #: ambiguity — it is a re-issuable transport failure.
    retryable = True

    def __init__(self, what: str, *, op_name: str) -> None:
        self.op_name = op_name
        super().__init__(what)
