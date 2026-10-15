"""
Base Registry Infrastructure

This module provides a base registry class for managing named registrations with
support for exact matching, regex pattern matching, and optional instance caching.

It also owns the registry KEY GRAMMAR (:func:`split_key` / :func:`split_dims` /
:func:`compose_key` / :func:`rebase_key`) and the :class:`RegistryKeyed` mixin
that binds a class hierarchy to one registry via ``class X(Base, key="…")``.

Usage:
    from lite.utils.registry import BaseRegistry

    # Define a typed registry
    class MyRegistry(BaseRegistry[MyBaseClass]):
        pass

    # Use as singleton via class methods
    MyRegistry.register("my-item", MyClass)
    instance = MyRegistry.get("my-item")

    # Or with regex patterns (auto-detected)
    MyRegistry.register(r"prefix@.*@suffix", MyClass)
    instance = MyRegistry.get("prefix@anything@suffix")

Run:
    python -m lite.utils.registry
"""

from __future__ import annotations

import dataclasses
import re
from itertools import product
from typing import Callable, ClassVar, Generic, Pattern, TypeVar

# type variable for the registered item type
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Key grammar — ``<name>[.<modifier>...]@<dim>[@<dim>...]``
#
# Owned here, next to the registries that resolve these keys, because the
# grammar is a DURABLE-RECORD contract and not a technique: an ``env_key`` is
# written into prompt-data parquet and read back by train, so every layer that
# splits or builds one must agree on where the ``@`` boundary is. The full
# specification of ``@`` vs ``.`` lives in :class:`BaseRegistry`'s "Key grammar"
# section.
# ---------------------------------------------------------------------------

def split_key(key: str) -> tuple[str, str]:
    """Split ``"<name>@<dims>"`` at the FIRST ``@``.

    Returns ``(name, dims)``. ``dims`` is the whole coordinate suffix with its
    own ``@`` separators intact (``"qwen3_vl@desktop@use"`` →
    ``("qwen3_vl", "desktop@use")``) and is the empty string when ``key``
    carries no ``@`` — some envs register without a task_id suffix (see
    :func:`lite.gym.registry.register`).

    Raises:
        ValueError: when ``key`` is empty or starts with ``@``.
    """
    name, _, dims = key.partition("@")
    if not name:
        raise ValueError(f"key {key!r} has empty name portion")
    return name, dims


def split_dims(key: str) -> list[str]:
    """The coordinate dims of ``key``, split out one per axis.

    ``split_dims("qwen3_vl@desktop@use")`` → ``["desktop", "use"]``; a key with
    no ``@`` yields ``[]``. This is the exact inverse of :func:`compose_key` —
    ``compose_key(split_key(k)[0], *split_dims(k)) == k`` for every ``k``
    :func:`split_key` accepts — and the reason callers that only want to scan the
    coordinate suffix (``@desktop`` / ``@point``) never need to know that the
    name sits at index 0 of a raw ``key.split("@")``.

    Raises:
        ValueError: when ``key`` has an empty name portion or empty dim entry.
    """
    _, dims = split_key(key)
    if not dims:
        if "@" in key:
            raise ValueError(f"key {key!r} has empty dim entry")
        return []
    split = dims.split("@")
    if any(not dim for dim in split):
        raise ValueError(f"key {key!r} has empty dim entry")
    return split


def compose_key(name: str, *dims: str) -> str:
    """Build ``"<name>@<dim>@<dim>..."`` from a name and its coordinate dims.

    Any ``.`` modifier already on ``name`` (``"qwen3_vl.base"``) binds to the
    name and rides ahead of the ``@`` suffix — the grammar's ordering rule.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"key name must be a non-empty string, got {name!r}")
    if "@" in name:
        raise ValueError(f"key name must not contain '@', got {name!r}")
    for dim in dims:
        if not isinstance(dim, str) or not dim:
            raise ValueError(f"key dim must be a non-empty string, got {dim!r}")
        if "@" in dim:
            raise ValueError(f"key dim must not contain '@', got {dim!r}")
    return "@".join((name, *dims))


def rebase_key(key: str, name: str) -> str:
    """Replace ``key``'s name, keeping its coordinate dims.

    ``rebase_key("qwen3_vl@desktop@use", "gpt")`` → ``"gpt@desktop@use"``.
    A key with no dims rebases to the bare ``name``.
    """
    return compose_key(name, *split_dims(key))


class BaseRegistry(Generic[T]):
    """
    Generic base class for registries with exact and regex pattern matching.

    This class provides a consistent API for:
    - Registering items by exact name or regex pattern
    - Retrieving items with automatic pattern matching
    - Optional instance caching
    - Decorator-based registration

    Subclasses should define their own class-level storage by overriding
    _init_storage() or by setting the class variables directly.

    Example:
        class MyRegistry(BaseRegistry[MyBaseClass]):
            # Each subclass gets its own storage automatically
            pass

        # Register with exact name
        MyRegistry.register("item-a", ItemA)

        # Register with regex pattern (auto-detected)
        MyRegistry.register(r"item@.*@type", ItemB)

        # Get instance
        instance = MyRegistry.get("item-a")
        instance = MyRegistry.get("item@foo@type")  # matches pattern

    Key grammar (``@`` vs ``.``)
    ----------------------------
    Keys across the agent/adapter/action-space/protocol registries follow::

        <name>[.<modifier>...]@<dim>@<dim>...

    - ``@`` separates *coordinate dimensions* — a value from a fixed, enumerable
      AXIS that locates the item in the addressing scheme:
      ``@<platform>`` (desktop/browser/mobile), ``@<task_type>`` (use /
      understanding / grounding.point / grounding.action / grounding.bbox),
      ``@<action_format>`` (point / bbox — NB distinct from the task_type
      ``grounding.bbox``).
      Most dims are composed at resolution time from ``env.metadata.dims`` —
      ``lite.agents.factory.make`` builds agent/adapter keys with
      ``compose_key(agent_id, *env.metadata.dims)`` — but some are fixed by the adapter
      rather than the env (the action-space ``@point`` / ``@bbox`` output
      format). Either way the segment is an axis value, not a free-form name.
      ``get()`` never splits on ``@``; it exact- or fullmatch-es the whole
      string, so ``@``'s meaning lives entirely in these composition /
      registration sites.
    - ``.`` attaches an author-fixed *modifier / sub-name* to the segment on its
      left — a free-form label, NOT an axis value: ``.base`` (a model's
      workflow-agnostic family), ``.history`` / ``.generic`` (a protocol
      flavor), ``.goal_image`` (an agent variant). (A ``.`` can also appear
      INSIDE an axis value — the task_type ``grounding.point`` sub-types the
      ``grounding`` axis.)
    - Ordering: dots bind to the name first (the full identity), ats trail as
      the coordinate suffix — ``qwen3_vl.base@browser@use`` (or, illustratively,
      the platform-specific ``qwen3_vl.history@mobile``).

    Rule of thumb: a value from a fixed structured axis (platform / task_type /
    action-format) takes ``@``; a free-form author-chosen name/label takes ``.``.

    Note: a bare ``.`` is treated as a LITERAL key character, not a regex
    wildcard (see ``_REGEX_METACHAR``), so literal-dot keys like ``qwen3_5.history``
    register as EXACT keys — they resolve by exact match and appear in ``list()``.
    A dot meant as a wildcard must be written explicitly as ``.*`` / ``.+``.
    """

    # Class-level storage - each subclass gets its own via __init_subclass__
    _items: ClassVar[dict[str, type[T] | Callable[..., T]]]
    _patterns: ClassVar[list[tuple[str, Pattern, type[T] | Callable[..., T]]]]
    _instances: ClassVar[dict[str, T]]

    # Regex metacharacters that indicate a pattern (not exact match). ``.`` is
    # deliberately EXCLUDED so a bare dot is a literal key char: literal-dot keys
    # (``qwen3_5.history``, ``grounding.point``) become EXACT keys, while genuine
    # patterns keep a real metachar (``qwen3_vl@(desktop|browser)@use``,
    # ``qwen3_vl\.base(@(desktop|browser|mobile)@use)?``). See the
    # "Key grammar" Note in the class docstring.
    _REGEX_METACHAR: ClassVar[Pattern] = re.compile(r"[*+?|\\()\[\]{}^$]")

    # Configuration: whether to cache instances by default
    _cache_by_default: ClassVar[bool] = True

    def __init_subclass__(cls, cache_by_default: bool = True, **kwargs) -> None:
        """
        Initialize storage for each subclass.

        Args:
            cache_by_default: Whether to cache instances by default
        """
        super().__init_subclass__(**kwargs)
        cls._items = {}
        cls._patterns = []
        cls._instances = {}
        cls._cache_by_default = cache_by_default

    @classmethod
    def _is_pattern(cls, name: str) -> bool:
        """Check if name contains regex metacharacters."""
        return bool(cls._REGEX_METACHAR.search(name))

    @classmethod
    def register(
        cls,
        name: str,
        item: type[T] | Callable[..., T],
        overwrite: bool = False,
    ) -> None:
        """
        Register an item.

        Args:
            name: Unique name or regex pattern for the item
            item: Class or factory function to register
            overwrite: Whether to overwrite existing registration

        Raises:
            ValueError: If name is already registered and overwrite=False

        Examples:
            # Exact match
            MyRegistry.register("my-item", MyClass)

            # Regex pattern (auto-detected by metacharacters)
            MyRegistry.register(r"prefix@.*@suffix", MyClass)
        """
        def _same_item(existing, new):
            """Check if existing and new items are effectively the same."""
            if existing is new:
                return True
            # For classes, check by qualified name only (handles module re-execution
            # where __module__ might differ due to runpy behavior)
            if isinstance(existing, type) and isinstance(new, type):
                return existing.__qualname__ == new.__qualname__
            return False
        
        if cls._is_pattern(name):
            # Check for existing pattern
            for i, (existing_name, _, existing_item) in enumerate(cls._patterns):
                if existing_name == name:
                    # Same item - skip silently (idempotent)
                    if _same_item(existing_item, item):
                        return
                    if not overwrite:
                        raise ValueError(
                            f"Pattern '{name}' is already registered. "
                            f"Use overwrite=True to replace it."
                        )
                    # Remove existing pattern
                    cls._patterns.pop(i)
                    break

            compiled = re.compile(name)
            cls._patterns.append((name, compiled, item))
        else:
            # Same item - skip silently (idempotent)
            if name in cls._items and _same_item(cls._items[name], item):
                return
            if name in cls._items and not overwrite:
                raise ValueError(
                    f"Item '{name}' is already registered. "
                    f"Use overwrite=True to replace it."
                )
            cls._items[name] = item

        # Clear cached instance if overwriting
        if name in cls._instances:
            del cls._instances[name]

    @classmethod
    def _find_item(
        cls, name: str
    ) -> tuple[type[T] | Callable[..., T], str | None] | None:
        """
        Find item by exact match or regex pattern match.

        Returns:
            tuple of (item, matched_pattern) or None if not found.
            matched_pattern is None for exact matches.
        """
        # First try exact match
        if name in cls._items:
            return cls._items[name], None

        # Then try regex pattern match — fullmatch covers concrete names
        # ("qwen3_5@browser@use" against pattern
        # "qwen3_5@(desktop|browser)@use"); the literal pattern-string
        # check covers self-lookup, where a class registered under a
        # pattern stores that pattern as its ``_registry_key`` and later
        # looks itself up with that exact string.
        for pattern_str, compiled, item in cls._patterns:
            if pattern_str == name or compiled.fullmatch(name):
                return item, pattern_str

        return None

    @classmethod
    def get(
        cls,
        name: str,
        *args,
        cache: bool | None = None,
        **kwargs,
    ) -> T:
        """
        Get an item instance by name.

        For dataclass items, kwargs are auto-separated: known dataclass
        fields are passed to the constructor, and unrecognized kwargs are
        packed into a ``kwargs`` catch-all field (if the class declares
        one). If there are unrecognized kwargs and no ``kwargs`` field,
        a ``TypeError`` is raised.

        Args:
            name: Name of the registered item (can match a pattern)
            *args: Arguments to pass to the constructor
            cache: Whether to cache the instance (defaults to class setting)
            **kwargs: Keyword arguments to pass to the constructor

        Returns:
            Item instance

        Raises:
            KeyError: If item is not registered
            TypeError: If unrecognized kwargs and no ``kwargs`` catch-all field
        """
        result = cls._find_item(name)
        if result is None:
            available = ", ".join(cls._items.keys()) or "none"
            patterns = ", ".join(p[0] for p in cls._patterns) or "none"
            raise KeyError(
                f"'{name}' not found in {cls.__name__}. "
                f"Available: {available}. Patterns: {patterns}"
            )

        item_factory, _matched_pattern = result

        # Auto-separate kwargs for dataclass items
        if kwargs and dataclasses.is_dataclass(item_factory):
            field_names = {f.name for f in dataclasses.fields(item_factory)}
            known_kw = {k: v for k, v in kwargs.items() if k in field_names}
            extra_kw = {k: v for k, v in kwargs.items() if k not in field_names}
            if extra_kw:
                if "kwargs" in field_names:
                    known_kw["kwargs"] = extra_kw
                else:
                    raise TypeError(
                        f"Unknown kwargs {set(extra_kw)} for "
                        f"{item_factory.__name__}. "
                        f"Available fields: {field_names}"
                    )
            kwargs = known_kw

        # Determine caching behavior
        should_cache = cache if cache is not None else cls._cache_by_default

        # Return cached instance if available and no args provided
        if should_cache and name in cls._instances and not args and not kwargs:
            return cls._instances[name]

        # Create new instance
        instance = item_factory(*args, **kwargs)

        # Cache if requested
        if should_cache and not args and not kwargs:
            cls._instances[name] = instance

        return instance

    @classmethod
    def get_class(
        cls, name: str
    ) -> type[T] | Callable[..., T]:
        """
        Get the registered class/factory without instantiation.

        Args:
            name: Name of the registered item (can match a pattern)

        Returns:
            The registered class or factory function

        Raises:
            KeyError: If item is not registered
        """
        result = cls._find_item(name)
        if result is None:
            available = ", ".join(cls._items.keys()) or "none"
            patterns = ", ".join(p[0] for p in cls._patterns) or "none"
            raise KeyError(
                f"'{name}' not found in {cls.__name__}. "
                f"Available: {available}. Patterns: {patterns}"
            )
        return result[0]

    @classmethod
    def list(cls) -> list[str]:
        """
        list all registered item names (excluding patterns).

        Returns:
            list of item names
        """
        return list(cls._items.keys())

    @classmethod
    def list_patterns(cls) -> list[str]:
        """
        list all registered patterns.

        Returns:
            list of pattern strings
        """
        return [p[0] for p in cls._patterns]

    @classmethod
    def list_expanded(cls) -> list[str]:
        """
        list all available keys, expanding alternation patterns.

        Exact keys are returned as-is.  Patterns that only use ``(a|b|c)``
        alternation groups are expanded into all concrete combinations via
        cartesian product.  Patterns with unexpandable wildcards (``.*``,
        ``+``, etc.) are skipped — convert them to explicit alternation first.

        Returns:
            Sorted list of concrete key strings.

        Example::

            # Given registrations:
            #   "item-a"                          (exact)
            #   r"prefix@(x|y)@suffix"            (expandable)
            #   r"other@.*@suffix"                (not expandable)
            MyRegistry.list_expanded()
            # → ["item-a", "prefix@x@suffix", "prefix@y@suffix"]
        """
        keys: list[str] = list(cls._items.keys())

        for pattern_str, _, _ in cls._patterns:
            expanded = cls._expand_pattern(pattern_str)
            if expanded is not None:
                keys.extend(expanded)

        return sorted(set(keys))

    @staticmethod
    def _expand_pattern(pattern: str) -> list[str] | None:
        """Expand a pattern with ``(a|b|c)`` groups into concrete keys.

        Returns ``None`` if the pattern contains unexpandable wildcards.
        """
        # Remove all (...) groups, then check remaining for regex metacharacters
        without_groups = re.sub(r"\([^)]*\)", "", pattern)
        if re.search(r"[.*+?\[\]{}^$\\]", without_groups):
            return None

        # Parse into segments: literals and alternation groups
        parts: list[list[str]] = []
        pos = 0
        for m in re.finditer(r"\(([^)]+)\)", pattern):
            if m.start() > pos:
                parts.append([pattern[pos : m.start()]])
            parts.append(m.group(1).split("|"))
            pos = m.end()
        if pos < len(pattern):
            parts.append([pattern[pos:]])

        if not parts:
            return [pattern]

        return ["".join(combo) for combo in product(*parts)]

    @classmethod
    def contains(cls, name: str) -> bool:
        """
        Check if an item is registered (exact or pattern match).

        Args:
            name: Name to check

        Returns:
            True if item is registered or matches a pattern
        """
        return cls._find_item(name) is not None

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister an item.

        Args:
            name: Name of the item to unregister

        Returns:
            True if item was unregistered, False if not found
        """
        removed = False

        if name in cls._items:
            del cls._items[name]
            removed = True

        # Also check patterns
        original_len = len(cls._patterns)
        cls._patterns = [(n, c, a) for n, c, a in cls._patterns if n != name]
        if len(cls._patterns) < original_len:
            removed = True

        if name in cls._instances:
            del cls._instances[name]

        return removed

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations and cached instances."""
        cls._items.clear()
        cls._patterns.clear()
        cls._instances.clear()


class RegistryKeyed:
    """Mixin that binds a class hierarchy to one :class:`BaseRegistry`.

    Every pillar whose subclasses declare their key inline —
    ``class Qwen3VLDesktopUseAdapter(BaseAgentAdapter, key="qwen3_vl@desktop@use")``
    — needs the same three things: the ``key=`` class keyword, registration of
    the class under it, and an accessor for it. This mixin owns all three.

    The hierarchy root names its registry once::

        class BaseAgentAdapter(RegistryKeyed, ABC):
            _registry = AgentAdapterRegistry

    A subclass defined WITHOUT ``key=`` is not registered and keeps its
    parent's key — deliberate: specialising a registered class does not create
    a new registry identity until a new key is given.
    """

    #: The registry this hierarchy registers into. Set on the hierarchy root.
    _registry: ClassVar[type[BaseRegistry]]

    #: The key this class was registered under; ``None`` when unregistered.
    _registry_key: ClassVar[str | None] = None

    def __init_subclass__(cls, key: str | None = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if key is not None:
            cls._registry.register(key, cls)
            cls._registry_key = key

    @classmethod
    def get_registry_key(cls) -> str | None:
        """The key this class is registered under, or ``None``."""
        return cls._registry_key


if __name__ == "__main__":
    print("Base Registry Infrastructure")
    print("=" * 40)

    # Demo: Create a simple registry
    class DemoItem:
        def __init__(self, value: str = "default"):
            self.value = value

    class DemoRegistry(BaseRegistry[DemoItem]):
        pass

    # Register items
    DemoRegistry.register("item-a", DemoItem)
    DemoRegistry.register(r"item@.*@type", DemoItem)

    print("\nRegistered items:", DemoRegistry.list())
    print("Registered patterns:", DemoRegistry.list_patterns())

    # Get instances
    a = DemoRegistry.get("item-a")
    print(f"\nitem-a: {a.value}")

    b = DemoRegistry.get("item@foo@type")
    print(f"item@foo@type (matches pattern): {b.value}")

    # Test caching
    a2 = DemoRegistry.get("item-a")
    print(f"\nCaching works: {a is a2}")
