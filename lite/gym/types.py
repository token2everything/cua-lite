"""
Gym-owned data types for cua-lite envs.

Reset/step envelopes, the ``info["executed_actions"]`` action trace, and
drift-reaper views live here. Shared protocol types are owned by
``lite.core.*`` and are imported from their owners at the use site, never
re-exported from this module. ``LiteSample`` is not a gym type and must not be
added here.

The one ``lite.core`` import below is an ORDINARY import, not a re-export: it
is the annotation of :attr:`LiteEnvStepResult.results`. The former metadata and
tool-call compatibility attributes are gone.

Run:
    uv run pytest tests/gym/types/test_envelope_contract.py -q
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from lite.core.tools.results import LiteToolResult

__all__ = [
    "EXECUTED_ACTIONS_INFO_KEY",
    "InstanceView",
    "LiteEnvObservation",
    "LiteEnvStepResult",
    "LiteExecutedAction",
    "ReapReport",
    "StateView",
]

# ---------------------------------------------------------------------------
# info["executed_actions"] — the env→consumer action trace
# ---------------------------------------------------------------------------

#: The ``LiteEnvStepResult.info`` key carrying the per-step action trace.
EXECUTED_ACTIONS_INFO_KEY = "executed_actions"


class _LiteExecutedActionOptional(TypedDict, total=False):
    args: dict[str, Any]


class LiteExecutedAction(_LiteExecutedActionOptional):
    """One action an env actually ran, as it appears in
    ``LiteEnvStepResult.info["executed_actions"]``.

    This is an ENV TRANSPORT shape, not a call: it is what the env DID, lowered
    into the backend's own vocabulary, and it is read back by the agent logger
    (``lite.agents.core.agent.logger``) and by trajectory-diff tooling. It is
    deliberately NOT :class:`~lite.core.tools.calls.LiteToolCall` — ``call`` may
    be a canonical action name (``left_click``), a synthetic ``noop``, or a raw
    backend string (``pyautogui.click(1918, 1078)``), none of which is a Lite
    tool name — which is exactly why it needs its own name instead of being
    described in a comment.

    ``args`` is optional: an env that lowers an action to a single opaque
    command string carries no structured arguments, and consumers already read
    it with a default.

    Every env that populates ``info[EXECUTED_ACTIONS_INFO_KEY]`` emits this
    shape, and the ``{"call", "args"}`` producers are annotated with it.

    The envs that pass a backend's trace straight through (``online_mind2web``,
    ``webharbor/webvoyager``, ``mobilegym``) DO conform on the declared keys —
    their servers build ``{"call": name, "args": args}`` — and they only ADD
    per-backend keys (``warning``, ``error``). That is a superset, not a second
    spelling, so consumers reading ``call``/``args`` are already correct on them.
    """

    call: str

# ---------------------------------------------------------------------------
# Observation / StepResult — env IO types
# ---------------------------------------------------------------------------

@dataclass
class LiteEnvObservation:
    """Initial environment observation returned by reset().

    Attributes:
        image: Raw PNG bytes of the initial visual observation, or None
            (text-only env or capture timeout). Base64 is deliberately NOT used as the in-memory
            representation — it is applied only at the one boundary that needs a
            text/data-URI form (the model adapter's image payload, off-loop).
            This keeps the image out of every JSON/base64 hop on the event
            loop and −25% in memory (raw PNG = 0.75× base64).
        text: Task instruction or initial text observation.
        metadata: Env-specific SMALL structured data (e.g. page_title, url).
            Embedded as a ``{"type": "metadata", "data": ...}`` content item
            in user messages; ignored by VL processors, used by protocols
            for rich history summaries. Must stay small: it is json-serialized
            into the wire-frame header (see /lite/gym/remote/frame.py). Large or
            binary payloads belong in ``image``/``text``, never here.
    """
    image: bytes | None
    text: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class LiteEnvStepResult:
    """Result returned by step(); model-visible feedback lives in results."""
    results: list[LiteToolResult] = field(default_factory=list)
    reward: float | None = None
    terminated: bool = False  # Task naturally ended (e.g. agent called terminate, eval says done)
    truncated: bool = False   # Cut off by external limit (e.g. max steps)
    # Optional env-specific metadata. Envs can put anything here; the one key
    # with a declared shape is ``EXECUTED_ACTIONS_INFO_KEY``, which carries a
    # ``list[LiteExecutedAction]`` (see that type for the per-env conformance
    # state).
    info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Drift-reaper contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstanceView:
    """One env-session as seen by the framework reconcile loop (which calls
    each env's ``live_ids``/``reap``).

    Frozen so a misbehaving env can't poison the snapshot. Fields are the
    minimum needed to reconstruct external-resource names; server-internal
    session fields (admission cost, raw bearer token, in-memory env object)
    stay hidden.
    """
    id: str                            # state.envs key — used in ReapReport.ghost_ids
    env_id: str                        # e.g. "androidworld", "lite.osworld"
    env_key: str                       # "{env_id}@{task_id}" used at construction
    session_id: str                    # cleanup/session label (NOT per-instance —
                                       # many instances of one job share this)
    token_hash: str | None             # per-client scope key (sha256(bearer)[:6]).
                                       # ``None`` only for legacy/unowned rows;
                                       # normal env-server instances always carry
                                       # a concrete owner hash.
    created_at: float                  # state clock; ghost-side age guard reads this
    external_resource_id: str | None   # opaque per-instance identifier, exact-match
                                       # key for orphan/ghost detection. ``None``
                                       # if env has no external resource yet
                                       # (e.g. between gym.make and first reset)


@dataclass(frozen=True)
class StateView:
    """Read-only snapshot of ``state.envs`` passed to the framework reconcile loop.

    Built once per reaper cycle under ``state.lock`` and used to derive each
    env's ``tracked`` set. Frozen so the loop can't mutate it.
    """
    by_env_id: dict[str, tuple[InstanceView, ...]]
    snapshot_at: float        # epoch seconds at snapshot time. Informational;
                              # hooks compute age via ``time.time()`` at hook
                              # execution + each ``InstanceView.created_at``.


@dataclass(frozen=True)
class ReapReport:
    """Result of one env's drift-reaper cycle.

    Asymmetry: env owns external-resource mutation (the hook removes
    its own orphans — e.g. ``docker rm -f`` for docker-backed envs,
    HTTP DELETE for service-pool envs — counted in
    ``orphans_reaped``); server owns ``state.envs`` mutation (it pops
    the ids in ``ghost_ids`` under its own lock).
    """
    orphans_reaped: int                  # external resources removed by THIS hook
    ghost_ids: tuple[str, ...] = ()      # state.envs ids whose external is gone
