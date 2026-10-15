"""Provider-independent action-space registry and base classes.

Action spaces own the model-facing tool schemas and the conversion boundary
between canonical Lite tool calls and a model family's tool-call format. They
do not execute tools; envs do that. They also do not own provider replay,
history, or runtime feedback policy; agents/adapters do that.

The important distinction is:

- tool layer: provider-visible schemas emitted by ``get_tool_schemas()``;
- action layer: canonical Lite actions gated by ``metadata.valid_actions``;
- native family layers: concrete model action-space modules own any provider
  enum spellings that are not canonical Lite action names.

Usage:
    @dataclasses.dataclass
    class MyActionSpace(BaseActionSpace, key="my_agent@desktop"):
        @staticmethod
        @tool(coordinate="(x, y) coordinates.")
        def click(coordinate: list[int]) -> dict[str, Any]:
            '''Perform a click.'''
            return make_tool_call("click", {"coordinate": coordinate})

    action_space = ActionSpaceRegistry.get("my_agent@desktop")
    schemas = action_space.get_tool_schemas()

Key grammar: action-space keys carry ``@<platform>`` and/or ``@<format>`` dims
(e.g. ``lite@desktop``, ``lite@point``, ``qwen3_vl@desktop@point``) — the
``@point`` / ``@bbox`` format is the coordinate-output axis the adapter selects,
so it uses ``@`` like ``@platform``. See the full ``@`` vs ``.`` grammar in
``lite.utils.registry.BaseRegistry``.

See ``lite/agents/models/README.md`` for model-family usage notes.
"""

from __future__ import annotations

import copy
import dataclasses
from abc import ABC
from typing import Any, ClassVar, final

from lite.core.errors import LiteContractError
from lite.core.messages import (
    get_action_description,
    get_inline_reasoning,
)
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import (
    LiteBBoxActionSet,
    LiteDesktopActionSet,
    LiteMobileActionSet,
    LitePointActionSet,
)
from lite.core.tools.calls import (
    LiteToolCall,
    make_tool_call,
    tool_call_arguments,
    tool_call_name,
)

# Re-exported for the public action-space facade; the catalog itself lives in
# ``lite.core.tools.extra_tools``.
from lite.core.tools.extra_tools import LiteBrowserNavToolSet  # noqa: F401
from lite.core.tools.schemas import (
    _compile_tool_method_schemas,
    tool_schema_name,
)
from lite.utils.registry import BaseRegistry, split_dims

__all__ = [
    "ActionSpaceRegistry",
    "BaseActionSpace",
    "LiteBBoxActionSpace",
    "LiteBrowserActionSpace",
    "LiteDesktopActionSpace",
    "LiteMobileActionSpace",
    "LitePointActionSpace",
    "assemble_tool_schemas",
]

#: One element of the text-render path: a tool-call dict in whatever shape THIS
#: class's ``format_tool_call_as_text`` accepts. Deliberately NOT a synonym for
#: ``LiteToolCall`` — the singular renderer owns that choice per class (canonical
#: on the base, a model family's own bare ``{name, arguments}`` projection
#: wherever that family overrides the renderer), and ``dict[str, Any]`` on the
#: plural signature would read as "any tool call works", which is the one thing
#: that is not true here.
TextRenderInput = dict[str, Any]

# =============================================================================
# Public API: Registry
# =============================================================================


class ActionSpaceRegistry(BaseRegistry[type["BaseActionSpace"]], cache_by_default=False):
    """Registry for action-space classes.

    Supports exact matching and regex pattern matching.
    Uses cache_by_default=False because action spaces may have instance state
    (e.g. resolution for coordinate conversion); callers pass kwargs to get().

    Example:
        ActionSpaceRegistry.get("lite@desktop")  # Get instance
        ActionSpaceRegistry.get_class("lite@desktop")  # Get class
        ActionSpaceRegistry.list()  # list registered keys
        ActionSpaceRegistry.list_patterns()  # list patterns
    """

    # Derived, never re-listed: ``LiteCUAMetadata.Platform`` is the one home of the
    # platform vocabulary, so a new platform reaches key parsing for free.
    _PLATFORM_SEGMENTS: ClassVar[frozenset[str]] = frozenset(
        platform.value for platform in LiteCUAMetadata.Platform
    )

    @classmethod
    def _platform_from_key(cls, name: str) -> str | None:
        """Return the concrete platform axis encoded in a registry key."""
        for segment in split_dims(name):
            if segment in cls._PLATFORM_SEGMENTS:
                return segment
        return None

    @classmethod
    def get(
        cls,
        name: str,
        *args,
        cache: bool | None = None,
        **kwargs,
    ) -> BaseActionSpace:
        instance = super().get(name, *args, cache=cache, **kwargs)
        platform = cls._platform_from_key(name)
        if platform is not None and getattr(instance, "platform", None) != platform:
            instance.platform = platform
        return instance


# =============================================================================
# Base Action Space Class
# =============================================================================


@dataclasses.dataclass
class BaseActionSpace(ABC):
    """Base class for model-facing action-space dialects.

    Three layers are deliberately separate:

    - **Tool**: what the provider sees. Use ``get_tool_names()``,
      ``get_tool_schemas()``, and ``get_tool_schema(name)``. An action-batch
      family may emit one action-batch tool while carrying many actions inside it.
    - **Action**: canonical Lite GUI actions carried by a tool call and gated by
      ``metadata.valid_actions``. Use ``get_action_names()``.

    Tool-layer invariants stay inside the tool layer:
    ``len(get_tool_names()) == len(get_tool_schemas())`` and
    ``get_tool_schema(n) is not None`` iff ``n in get_tool_names()``.
    No invariant crosses from emitted provider tools to canonical carried
    actions.

    An action space defines provider schemas, text formatting helpers, and
    conversion between canonical Lite calls and model-specific calls.

    Example:
        @dataclasses.dataclass
        class MyActionSpace(BaseActionSpace, key="my_agent@desktop"):
            @staticmethod
            @tool(coordinate="(x, y) coordinates.")
            def click(coordinate: list[int]) -> dict[str, Any]:
                '''Perform a click.'''
                return make_tool_call("click", {"coordinate": coordinate})

    Attributes:
        platform: Platform type ("desktop", "browser", or "mobile")
    """

    platform: str = "desktop"

    # -- Class-level cache for schemas (populated by __init_subclass__) --
    _SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {}
    _DEFAULTS: ClassVar[dict[str, dict[str, Any]]] = {}
    _registry_key: ClassVar[str | None] = None

    # -------------------------------------------------------------------------
    # Public Methods: Schema Access
    # -------------------------------------------------------------------------

    @classmethod
    def get_tool_schemas(cls, include: list[str] | None = None) -> list[dict[str, Any]]:
        """Get canonical Lite tool schemas emitted by this action space.

        ``include`` is an action-space filter: unmatched names are dropped. Some
        grounding spaces forward includes across vocabularies, so this differs
        from the tool-catalog selector semantics on ``BaseTools``.

        Args:
            include: If set, only return schemas for these action names.

        Returns:
            list of tool schema dicts (deep copy). Each element is one tool's schema.
        """
        schemas = list(cls._SCHEMAS.values())
        if include is not None:
            include_set = set(include)
            schemas = [s for s in schemas if tool_schema_name(s) in include_set]
        return copy.deepcopy(schemas)

    @classmethod
    def get_tool_names(cls) -> frozenset[str]:
        """Names of the tools this space EMITS — derived from ``get_tool_schemas()``.

        ``get_tool_schemas()`` is the single root of the tool layer; the other
        two accessors are derived from it so both tool-layer invariants
        (``len(names) == len(schemas)`` and ``get_tool_schema(n) is not None
        <=> n in get_tool_names()``) hold by construction, including on
        action-batch spaces whose whole surface collapses into one tool and on
        opaque-provider spaces that emit nothing at all.

        This is NOT the canonical action question: use ``get_action_names()``
        for the actions an action-batch space carries inside its top-level tool.
        """
        return frozenset(tool_schema_name(s) for s in cls.get_tool_schemas())

    @classmethod
    def get_declared_action_schema_names(cls) -> frozenset[str]:
        """Schema names this space declares before emitted-schema filtering.

        ``valid_actions`` uses these names to decide which declared action or
        action-batch child schema it owns. Model-native enum spellings, when a
        family has them, are declared and filtered by that family's action-space
        module.
        """
        return frozenset(cls._SCHEMAS)

    @classmethod
    def get_tool_schema(cls, name: str) -> dict[str, Any] | None:
        """Get one EMITTED tool schema by name — derived from ``get_tool_schemas()``.

        Scans the UNFILTERED list on purpose: ``include=`` is an action-layer
        filter on action-batch spaces and a tool-layer filter on flat ones, so
        ``get_tool_schemas(include=[name])`` cannot answer this question.

        Returns:
            The tool schema dict, or None if this space emits no such tool.
        """
        for schema in cls.get_tool_schemas():
            if tool_schema_name(schema) == name:
                return schema
        return None

    @classmethod
    def get_registry_key(cls) -> str | None:
        """Get the registry key for this action space class."""
        return cls._registry_key

    # -------------------------------------------------------------------------
    # Public Methods: Text Format Support
    # -------------------------------------------------------------------------

    def format_tool_call_as_text(self, tool_call: LiteToolCall) -> str:
        """Render one CANONICAL Lite tool call as ``name(key=value, ...)``.

        The declared parameter type IS the contract, not an implementation
        detail: the base reads through :func:`tool_call_name` /
        :func:`tool_call_arguments`, so it requires the nested ``function``
        envelope and raises ``KeyError`` on a bare ``{name, arguments}``
        projection. This method is the single place a class decides that shape;
        overriding families that render their own bare projection re-declare it
        here, and a caller must know which class it holds before choosing what
        to pass.
        """
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)

        parts = []
        for k, v in args.items():
            if isinstance(v, str):
                parts.append(f"{k}='{v}'")
            else:
                parts.append(f"{k}={v!r}")

        return f"{name}({', '.join(parts)})"

    @final
    def format_tool_calls_as_text(self, tool_calls: list[TextRenderInput]) -> str:
        """Join THIS class's :meth:`format_tool_call_as_text` element-wise.

        Declares no input shape of its own — hence ``TextRenderInput`` rather
        than a concrete call type, and hence ``@final``: the shape is decided
        once, by whichever :meth:`format_tool_call_as_text` the class binds, and
        a family must not add a second decision here. Elements are canonical
        Lite calls for the base and the family's own projection for a family
        that overrides the singular renderer.
        """
        return "\n\n".join(self.format_tool_call_as_text(tc) for tc in tool_calls)

    def format_message_as_text(self, message: dict[str, Any]) -> str:
        """Convert an assistant message to text, embedding reasoning and tool_calls.

        Output format (each part omitted when its source is empty)::

            Thought: <inline_reasoning>
            <action_description>
            Action: name(key=value)

        ``Thought:`` is sourced from the prompted ``InlineReasoningContent``
        part(s) via :func:`get_inline_reasoning` (strict inline-only; does
        NOT merge the NATIVE ``reasoning_content`` slot — that channel is
        rendered by the chat_template's Thinking-mode branch, not here).
        The ``<action_description>`` line is the prose describing the
        action and is meaningful only on turns with ``tool_calls``; no-tool
        final prose stays as ``TextContent`` and is preserved by adapter-level
        content-only branches. It is never re-labeled as ``Thought:``.
        Subclasses may override for agent-specific formatting.
        """
        parts = []

        reasoning = get_inline_reasoning(message)
        action_description = get_action_description(message)

        if reasoning:
            parts.append(f"Thought: {reasoning}")
        if action_description:
            parts.append(action_description)

        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            action_text = self.format_tool_calls_as_text(tool_calls)
            parts.append(f"Action: {action_text}")

        return "\n".join(parts)

    # -------------------------------------------------------------------------
    # Public Methods: Tool Call Conversion (must implement)
    # -------------------------------------------------------------------------

    def convert_tool_calls_to_agent(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        resolution: tuple[int, int] | None = None,
        **kwargs,
    ) -> list[Any]:
        """
        Convert CUA-lite tool calls to agent's format.

        Default: project to the provider-independent bare model-function shape
        (``{name, arguments}``) used by Lite-style adapters at the adapter
        boundary. This is not canonical storage: persisted Lite calls keep the
        nested ``function`` envelope. Subclasses with a divergent agent format
        (e.g. provider-native ``computer_use`` / ``mobile_use`` wrappers or
        provider-native computer actions) override.

        Note: list-to-list because one action may map to multiple or vice versa.

        The render direction intentionally has a narrower named surface than
        the parse direction below. It currently only advertises ``resolution``;
        ``**kwargs`` remains for family-specific extension points.

        Args:
            tool_calls: CUA-lite tool call dicts
            resolution: ``(width, height)`` of the frame the agent's
                coordinates live in. MANDATORY for the families that change
                coordinate frame (claude, gpt) and unused by the rest; the
                requirement is per-family and raised there, not here.
            **kwargs: family-specific extras.

        Returns:
            Tool calls in agent's format
        """
        result: list[Any] = []
        for tool_call in tool_calls:
            result.extend(self._convert_single_to_agent(tool_call))
        return result

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[Any]:
        """Render one canonical call into this family's agent projection.

        Families with per-call provider projections override this hook and
        inherit the plural fan-out in :meth:`convert_tool_calls_to_agent`.
        Families that need extra per-call context, such as a coordinate
        resolution, override the plural entry point directly.
        """
        return [
            {
                "name": tool_call_name(tool_call),
                "arguments": copy.deepcopy(tool_call_arguments(tool_call)),
            }
        ]

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[Any],
        *,
        resolution: tuple[int, int] | None = None,
        active_extra_tool_names: set[str] | None = None,
        active_extra_tool_schemas: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Convert agent's tool calls to CUA-lite format.

        Default: wrap bare model-function projections (``{name, arguments}``)
        as pre-stamp canonical Lite calls. Subclasses override for agent
        formats that diverge from CUA-lite.

        The parse direction is intentionally wider than the render direction:
        it must decide what an unrecognised tool name means using the active
        tool surface.

        Every rejection below is a CALLER-code violation, not malformed model
        output, so it stays a bare ``TypeError`` / ``ValueError`` instead of the
        parse-boundary-owned ``ModelToolCallParseError``: the parser that calls
        this builds the bare projection itself, so a non-dict element, a
        provider ``call_id`` smuggled onto a GUI call, or a canonical nested
        envelope arriving on the parse side is a layer mistake in the caller.
        Spelling them as model-output errors would let
        ``AdapterBasedAgent._parse_generation_response()`` rewrite an adapter bug
        into a terminal model parse-failure final. A family whose wire grammar
        lets the MODEL choose these shapes owns that judgement in its own
        override.

        Args:
            agent_tool_calls: Tool calls in agent's format
            resolution: ``(width, height)`` of the frame the agent's
                coordinates arrive in.
            active_extra_tool_names: the standalone tools this sample actually
                advertises. Parsers may use this to resolve provider spellings
                that depend on the active surface; runtime env validation remains
                the authority for unsupported calls.
            active_extra_tool_schemas: their full schemas, so an env-owned call
                is admitted on schema satisfaction rather than name alone.
            **kwargs: family-specific extras.

        Returns:
            CUA-lite tool call dicts
        """
        result: list[dict[str, Any]] = []
        for i, agent_tool_call in enumerate(agent_tool_calls):
            where = f"agent_tool_calls[{i}]"
            if not isinstance(agent_tool_call, dict):
                raise TypeError(f"{where} must be a bare tool-call dict")
            extra_keys = sorted(set(agent_tool_call) - {"name", "arguments"})
            if extra_keys:
                raise ValueError(f"{where} has non-bare-call keys {extra_keys}")
            try:
                name = agent_tool_call["name"]
                arguments = agent_tool_call["arguments"]
            except KeyError as exc:
                raise ValueError(f"{where} must be bare {{name, arguments}}") from exc
            if not isinstance(name, str) or not name:
                raise ValueError(f"{where}.name must be a non-empty string")
            if not isinstance(arguments, dict):
                raise ValueError(f"{where}.arguments must be a dict")
            # Those five checks are exactly the domain constraints
            # ``validate_lite_tool_call`` would re-derive, so the constructed
            # pre-stamp call is canonical by construction.
            result.append(make_tool_call(name, copy.deepcopy(arguments)))
        return result

    # -- Valid-action filtering ----------------------------------------------
    #: Canonical action that admits this space's owned schema under
    #: ``metadata.valid_actions`` when the emitted schema name differs from the
    #: canonical action name (for example a grounding ``click`` schema admitted
    #: by ``valid_actions=["point"]``).
    _LITE_FORMAT_ACTION_NAME: ClassVar[str | None] = None

    @classmethod
    def filter_tool_schemas_for_valid_actions(
        cls,
        schemas: list[dict[str, Any]],
        valid_actions: list[str],
    ) -> list[dict[str, Any]]:
        """Filter provider tool schemas using canonical action-space ownership.

        Base owns only one-schema-per-tool canonical schema gating: a schema
        this space declares is kept when its canonical action is allowed, and
        every other schema passes through untouched — including a schema with
        no canonical Lite name at all. Provider wrapper enums and opaque native
        computer schemas therefore stay owned by the concrete model-family
        action spaces that know those wire strings.
        """
        allowed = set(valid_actions)
        owned_schemas = cls.get_declared_action_schema_names()
        format_action = cls._LITE_FORMAT_ACTION_NAME
        result: list[dict[str, Any]] = []
        for schema in copy.deepcopy(schemas):
            try:
                schema_name = tool_schema_name(schema)
            except LiteContractError:
                # No canonical Lite name => not a schema this space declares.
                result.append(schema)
                continue
            if schema_name not in owned_schemas:
                result.append(schema)
                continue
            valid_action = format_action or schema_name
            if valid_action in allowed:
                result.append(schema)
        return result

    # -------------------------------------------------------------------------
    # Internal: Schema Pre-computation (runs once at class definition time)
    # -------------------------------------------------------------------------

    def __init_subclass__(cls, key: str | None = None, **kwargs):
        """
        Pre-compute and cache tool schemas from @tool decorated methods.

        This runs ONCE when the class is defined (not on each instantiation).
        It parses type hints and docstrings to generate JSON Schemas, storing
        them in cls._SCHEMAS for fast access via get_tool_schemas().

        Args:
            key: Optional registry key to auto-register this action space

        Why here instead of get_tool_schemas()?
        - Runs once at import time, not on every call
        - Catches type annotation errors early
        - Subclasses automatically inherit parent schemas
        """
        super().__init_subclass__(**kwargs)

        # Auto-register if key is provided
        if key is not None:
            ActionSpaceRegistry.register(key, cls)
            cls._registry_key = key

        # Initialize fresh storage for this class
        cls._SCHEMAS = {}
        cls._DEFAULTS = {}

        # Inherit schemas from parent classes
        for base in cls.__mro__[1:]:
            if hasattr(base, "_SCHEMAS") and base._SCHEMAS:
                for name, schema in base._SCHEMAS.items():
                    if name not in cls._SCHEMAS:
                        cls._SCHEMAS[name] = copy.deepcopy(schema)
            if hasattr(base, "_DEFAULTS") and base._DEFAULTS:
                for name, defaults in base._DEFAULTS.items():
                    if name not in cls._DEFAULTS:
                        cls._DEFAULTS[name] = copy.deepcopy(defaults)

        # Generate schemas from @tool decorated methods in THIS class.
        schemas, defaults = _compile_tool_method_schemas(cls)
        cls._SCHEMAS.update(schemas)
        cls._DEFAULTS.update(defaults)


# =============================================================================
# CUA-Lite Desktop Action Space
# =============================================================================


@dataclasses.dataclass
class LiteDesktopActionSpace(LiteDesktopActionSet, BaseActionSpace, key=r"lite@desktop"):
    """
    Desktop action space with normalized [0, 1000] coordinates.

    Example:
        desktop = LiteDesktopActionSpace()
        tools = desktop.get_tool_schemas()
        action = LiteDesktopActionSpace.click(coordinate=[500, 300])
    """

    platform: str = "desktop"

    @classmethod
    def filter_tool_schemas_for_valid_actions(
        cls,
        schemas: list[dict[str, Any]],
        valid_actions: list[str],
    ) -> list[dict[str, Any]]:
        if any(
            tool_schema_name(schema) == cls._ACTION_BATCH_TOOL_NAME
            for schema in schemas
        ):
            return cls.filter_child_action_enum(schemas, valid_actions)
        return super().filter_tool_schemas_for_valid_actions(schemas, valid_actions)


# =============================================================================
# CUA-Lite Browser Action Space
# =============================================================================

# The browser-universal nav/tab tools are extra tools selected by env metadata.
# ``lite@browser`` is only a registered desktop-action reference; it must not become
# a hidden source of default nav verbs.


@dataclasses.dataclass
class LiteBrowserActionSpace(LiteDesktopActionSpace, key=r"lite@browser"):
    """The browser action space: desktop coordinate verbs registered under
    ``lite@browser``.

    **Browser nav tools are surfaced as ``extra_tools``, never as action-space verbs
    the agent emits directly** — the ``lite@(desktop|browser)@use`` adapter
    uses desktop-coordinate action verbs, and per-model action spaces register
    browser keys to their desktop-coordinate provider classes. This generic
    ``lite@browser`` entry is the reference browser surface; nav tool schemas are
    selected only through resolved
    ``metadata.extra_tool_schemas``."""

    platform: str = "browser"


# =============================================================================
# CUA-Lite Mobile Action Space
# =============================================================================


@dataclasses.dataclass
class LiteMobileActionSpace(LiteMobileActionSet, BaseActionSpace, key="lite@mobile"):
    """
    Mobile action space with normalized [0, 1000] coordinates.

    Example:
        mobile = LiteMobileActionSpace()
        tools = mobile.get_tool_schemas()
        action = LiteMobileActionSpace.tap(coordinate=[500, 300])
    """

    platform: str = "mobile"

    @classmethod
    def filter_tool_schemas_for_valid_actions(
        cls,
        schemas: list[dict[str, Any]],
        valid_actions: list[str],
    ) -> list[dict[str, Any]]:
        if any(
            tool_schema_name(schema) == cls._ACTION_BATCH_TOOL_NAME
            for schema in schemas
        ):
            return cls.filter_child_action_enum(schemas, valid_actions)
        return super().filter_tool_schemas_for_valid_actions(schemas, valid_actions)


# =============================================================================
# Grounding Action Spaces
# =============================================================================


@dataclasses.dataclass
class LitePointActionSpace(LitePointActionSet, BaseActionSpace, key="lite@point"):
    """Platform-neutral point-grounding action space.

    The registry key has no platform segment, so the metadata default remains
    ``"desktop"``. Per-platform model grounding spaces carry the platform in
    their own keys.
    """

    platform: str = "desktop"
    _LITE_FORMAT_ACTION_NAME = next(iter(LitePointActionSet.get_action_names()))


@dataclasses.dataclass
class LiteBBoxActionSpace(LiteBBoxActionSet, BaseActionSpace, key="lite@bbox"):
    """Bounding box grounding action space - locate objects by bounding box.

    ``platform`` is a DECLARED ``str`` for the same reason as
    :class:`LitePointActionSpace` — see its docstring.
    """

    platform: str = "desktop"
    _LITE_FORMAT_ACTION_NAME = next(iter(LiteBBoxActionSet.get_action_names()))


def assemble_tool_schemas(
    action_space: BaseActionSpace,
    base_schemas: list[dict[str, Any]],
    *,
    valid_actions: list[str] | None,
    extra_tool_schemas: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """The one tool-assembly pipeline: gate, then merge the env's extras.

    A free function, not a method on either side. Tool assembly is shared by
    adapter-backed agents and direct API agents, so it cannot live on the
    adapter. It is not a method on the action space either: ``base_schemas``
    comes from the CALLER, because an API agent's provider-native schema needs
    runtime display dimensions and a tool-version config that the action space
    never sees.

    There is deliberately NO ``metadata`` parameter. The two fields it would
    carry travel as arguments, which is exactly what lets an agent (no adapter,
    hence no ``self.metadata``) and an adapter share one pipeline.

    Args:
        action_space: the space whose class owns both gates.
        base_schemas: what this caller starts from — the space's own
            ``get_tool_schemas()`` for an adapter family, a provider-native
            schema for an API agent.
        valid_actions: ``metadata.valid_actions``. ``None`` skips the gate
            entirely, which is DISTINCT from ``[]`` (block everything).
        extra_tool_schemas: ``metadata.extra_tool_schemas``. Appended as the
            standalone tool surface this sample offers.

    Provider action-value pruning is NOT done here. A family that hides extra
    tools behind its own wrapper ``action`` enum owns that gate and calls it from
    its own adapter, because the mapping from a provider action value to a Lite
    extra tool is family policy, not a base-class fact.
    """
    cls = type(action_space)
    schemas = base_schemas
    if valid_actions is not None:
        schemas = cls.filter_tool_schemas_for_valid_actions(schemas, valid_actions)
    if extra_tool_schemas:
        schemas = schemas + list(extra_tool_schemas)
    return schemas
