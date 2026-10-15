"""
Abstract base class for CUA-Lite environments — the caller-facing interface.

:class:`LiteBaseEnv` is the whole surface an agent / rollout driver uses to run
an env (``metadata`` / ``reset`` / ``step`` / ``close``), identical in direct and
env-server modes. It is implemented by concrete envs, by ``EnvWrapper``, and by
the remote proxy ``LiteEnvClient``.

Env-server integration lives OUTSIDE this base so the caller surface stays clean
(see :mod:`lite.gym.services`):
  * explicit construction + bind/boot lifecycle → :class:`lite.gym.services.EnvServerPoolable`
  * drift-reaper resource id → :class:`lite.gym.services.EnvServerResource`

See /docs/envs.md for the env-author guide.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from lite.core.metadata import LiteBaseMetadata
from lite.core.tools.calls import RuntimeEnvAction
from lite.core.tools.schemas import BaseTools, LiteToolSchema
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult

if TYPE_CHECKING:
    from lite.gym.utils.config.identity import EnvIdentity


class LiteBaseEnv(ABC):
    """The caller-facing CUA-Lite environment interface.

    The whole surface an agent / rollout driver uses; identical in direct and
    env-server modes.

    Subclasses MUST implement:
      * :meth:`_runtime_metadata` — task-level metadata (see the same-source
        contract below; the final :meth:`metadata` property is framework-owned)
      * :meth:`reset` — async, returns initial observation
      * :meth:`step` — async, executes actions
      * :meth:`close` — async, releases resources

    Subclasses define their own ``__init__`` freely.

    Env-server integration is opt-in and lives OUTSIDE this base (so the caller
    surface stays uncluttered):
      * explicit construction + bind/boot lifecycle → inherit :class:`lite.gym.services.EnvServerPoolable`
      * drift-reaper resource id → mix in :class:`lite.gym.services.EnvServerResource`
    """

    #: Server-injected per-instance context (session / token / port), set by
    #: :func:`lite.gym.registry.make` in server mode; ``None`` in direct mode.
    identity: EnvIdentity | None = None

    #: Registry-key identity, set by ``gym.make`` after construction. ``None``
    #: on a directly-constructed env — the identity merge below then skips.
    #: RESERVED: env code never writes these (see ``LiteBaseMetadata.others``).
    env_id: str | None = None
    task_id: str | None = None

    #: The env<->agent tool contract, half one: what this env DECLARES.
    #: A ``BaseTools`` subclass — the same base the core sets
    #: (``LiteFinishToolSet`` / ``LiteBrowserNavToolSet`` / ``LiteAppLaunchToolSet``) and the GUI
    #: action sets hang off, so "which tools does X expose" is one question with
    #: one kind of answer on both sides of the boundary. The class is the
    #: AUTHORING form only: what travels in ``metadata.extra_tool_schemas`` is
    #: the canonical Lite schemas :meth:`extra_tool_schemas` compiles out of it.
    #:
    #: Canonical finish tools are NEVER declared here (the resolver raises) —
    #: they are resolved centrally — which is why every "is this a name this env
    #: knows" predicate must spell ``EXTRA_TOOLS.get_tool_names() |
    #: LiteFinishToolSet.get_tool_names()`` and never the bare accessor.
    #: Default: the empty set, i.e. finish tools only.
    EXTRA_TOOLS: ClassVar[type[BaseTools]] = BaseTools

    #: Half two, and deliberately SEPARATE: what the backend is wired to
    #: EXECUTE. Orthogonal to the declaration — a name can be declared and still
    #: be rejected because the transport cannot honor it (``bash`` is gated by
    #: transport family rather than env identity, and is never declared by any
    #: env). ``None`` = everything declared is executable.
    EXECUTABLE_EXTRA_TOOLS: ClassVar[frozenset[str] | None] = None

    @classmethod
    def extra_tool_schemas(
        cls,
        selection: list[str] | None = None,
        **ctor_args: Any,
    ) -> list[LiteToolSchema]:
        """Resolve ``env_kwargs.extra_tools`` → canonical Lite tool schemas.

        A CLASSMETHOD, not an attribute, and not abstract. Not an attribute
        because an env's tool surface is not purely class-level: browsergym
        derives its set by introspecting a third-party action set chosen by a
        ctor arg, and the sandbox family merges a per-TASK override into the
        class default. Not an instance method because every env resolves this
        at REGISTRATION time, with no instance — the metadata builders are
        ``@staticmethod``s or bare module functions. Not abstract because the
        class-declared default below is the right answer for most envs; the
        ones that must derive override it and consume ``ctor_args``.
        """
        from lite.gym.utils.feedback.surface import resolve_extra_tools

        return resolve_extra_tools(
            selection,
            tools=cls.EXTRA_TOOLS,
            env_name=cls.__name__,
            executable=cls.EXECUTABLE_EXTRA_TOOLS,
        )

    @classmethod
    def known_standalone_tool_names(cls) -> frozenset[str]:
        """Standalone names this env KNOWS — declared extras plus finish.

        The single spelling of the inactive-vs-unknown question. The
        ``| LiteFinishToolSet.get_tool_names()`` half is not optional: an env's ``EXTRA_TOOLS``
        structurally cannot contain ``response``/``terminate``, so the bare
        accessor would silently classify both as *unknown* calls.
        """
        from lite.core.tools.extra_tools import LiteFinishToolSet

        return cls.EXTRA_TOOLS.get_tool_names() | LiteFinishToolSet.get_tool_names()

    @property
    def metadata(self) -> LiteBaseMetadata:
        """Task-level metadata, available immediately after construction.

        FRAMEWORK-OWNED: the value is the
        env's :meth:`_runtime_metadata` — itself ``_task_metadata(record) ⊕
        env_kwargs amendments`` — plus the registry-key identity merged into
        ``others`` when this env was constructed via ``gym.make``. Env authors
        implement ``_task_metadata`` + ``_runtime_metadata``, never this
        property (enforced by ``__init_subclass__``; the two framework
        classes opt out with ``allow_metadata_override=True``).

        Implementations of ``_runtime_metadata`` MUST be pure functions of
        constructor/bind args (and module-level constants) — no dependence on
        state that only exists after :meth:`reset` (browser pages, docker
        containers, ...), and no direct ``os.environ`` reads. This invariant
        lets :class:`lite.gym.remote.LiteEnvClient` eager-create the
        server-side instance during construction so its ``metadata`` is live
        by the time ``gym.make()`` returns.
        """
        md = self._runtime_metadata()
        if self.env_id is None:
            return md
        return dataclasses.replace(md, others={
            **md.others,
            "env_id": self.env_id,
            "task_id": self.task_id,
        })

    @abstractmethod
    def _runtime_metadata(self) -> LiteBaseMetadata:
        """Env-owned metadata hook: ``_task_metadata(…) ⊕ env_kwargs amendments``.

        Abstract so an env that forgets it fails AT INSTANTIATION (the
        pre-invariant behavior of the abstract ``metadata`` property).
        ``_task_metadata`` itself is not declared here because its arity is
        env-specific.
        """
        ...

    def __init_subclass__(cls, *, allow_metadata_override: bool = False, **kwargs) -> None:
        """Enforce the same-source contract: env authors implement
        ``_task_metadata`` + ``_runtime_metadata``, never the final
        ``metadata`` property. The PEP-487 keyword opt-out is for the two
        framework classes only (``EnvWrapper`` pass-through, ``LiteEnvClient``
        server-computed) and deliberately
        does NOT inherit — a subclass of an exempted class trips the guard.
        """
        super().__init_subclass__(**kwargs)
        if not allow_metadata_override and "metadata" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} overrides `metadata` — implement "
                "`_task_metadata` + `_runtime_metadata` instead; "
                "framework classes may opt out with allow_metadata_override=True"
            )

    @abstractmethod
    async def reset(self) -> LiteEnvObservation:
        """Reset the environment and return initial observation."""
        ...

    @abstractmethod
    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        """Execute actions and return per-call results + terminal state.

        ``actions`` is the canonical ``{"type", "function"}`` call envelope,
        which may also carry the runtime sidecars for internal finishes and
        result-call routing. Concrete envs lower it to the env-internal bare
        action through ``prepare_env_tool_calls``.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release resources. After ``close()`` this instance is discarded."""
        ...

    @property
    def unwrapped(self) -> LiteBaseEnv:
        """Innermost env (no-op default; overridden by ``EnvWrapper``).

        Lets callers reach members past wrapper layers — ``__getattr__``
        delegation forwards methods but does NOT survive ``isinstance``. Mirrors
        gymnasium's ``Env.unwrapped``; it is also how the env-server introspects
        an env's capabilities (e.g. ``isinstance(env.unwrapped, EnvServerResource)``).
        """
        return self

    async def __aenter__(self) -> LiteBaseEnv:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
