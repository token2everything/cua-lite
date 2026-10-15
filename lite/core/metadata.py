"""Task-level Lite metadata contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

from lite.core.errors import LiteContractError
from lite.core.tools.schemas import LiteToolSchema, validate_extra_tool_schemas


def _coerce_closed_enum(field_name: str, enum_cls: type[StrEnum], value: Any) -> StrEnum:
    """Coerce a wire value into a closed metadata enum."""
    try:
        return enum_cls(value)
    except ValueError:
        legal = ", ".join(member.value for member in enum_cls)
        raise LiteContractError(
            f"metadata.{field_name} must be one of [{legal}]; got {value!r}"
        ) from None


def _required_keys(data: dict[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise LiteContractError(f"metadata is missing required keys {missing}")


def _reject_unknown_keys(data: dict[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise LiteContractError(f"metadata has unknown keys {unknown}")


def _require_metadata_kind(data: dict[str, Any], expected: str) -> None:
    _required_keys(data, ("metadata_kind", "dims"))
    if data["metadata_kind"] != expected:
        raise LiteContractError(
            f"metadata.metadata_kind must be {expected!r}; got {data['metadata_kind']!r}"
        )


@dataclass(kw_only=True, slots=True)
class LiteBaseMetadata(ABC):
    """Base task metadata shared by Lite task contracts.

    ``metadata_kind`` selects the semantic contract. ``dims`` is only the
    routing coordinate tuple consumed by generic registries and training loops.
    """

    metadata_kind: ClassVar[str]

    dims: tuple[str, ...] = ()
    extra_tool_schemas: list[LiteToolSchema] = field(default_factory=list)
    others: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.dims, (list, tuple)):
            raise LiteContractError("metadata.dims must be a list or tuple")
        dims = tuple(self.dims)
        for dim in dims:
            if not isinstance(dim, str):
                raise LiteContractError("metadata.dims entries must be strings")
            if not dim:
                raise LiteContractError("metadata.dims entries must be non-empty")
            if "@" in dim:
                raise LiteContractError("metadata.dims entries must not contain '@'")
        self.dims = dims

        if not isinstance(self.extra_tool_schemas, list):
            raise LiteContractError("metadata.extra_tool_schemas must be a list")
        self.extra_tool_schemas = list(self.extra_tool_schemas)
        validate_extra_tool_schemas(self.extra_tool_schemas)

        if not isinstance(self.others, dict):
            raise LiteContractError("metadata.others must be a dict")
        self.others = dict(self.others)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to the canonical row shape."""


@dataclass(kw_only=True, slots=True)
class LiteCUAMetadata(LiteBaseMetadata):
    """Computer-use task metadata.

    ``dims`` is always ``(<platform>, <task_type>)``. ``platform``,
    ``task_type``, and ``valid_actions`` are CUA-only semantics, not generic
    routing fields.
    """

    class Platform(StrEnum):
        DESKTOP = "desktop"
        BROWSER = "browser"
        MOBILE = "mobile"

    class TaskType(StrEnum):
        UNDERSTANDING = "understanding"
        GROUNDING_ACTION = "grounding.action"
        GROUNDING_POINT = "grounding.point"
        GROUNDING_BBOX = "grounding.bbox"
        USE = "use"

    metadata_kind: ClassVar[str] = "cua"

    dims: tuple[str, ...] = (Platform.DESKTOP.value, TaskType.USE.value)
    valid_actions: list[str] | None = None

    def __post_init__(self) -> None:
        LiteBaseMetadata.__post_init__(self)
        if len(self.dims) != 2:
            raise LiteContractError("metadata.dims for CUA metadata must have two entries")
        platform = _coerce_closed_enum("dims[0]", self.Platform, self.dims[0])
        task_type = _coerce_closed_enum("dims[1]", self.TaskType, self.dims[1])
        self.dims = (platform.value, task_type.value)

        if self.valid_actions is not None:
            if not isinstance(self.valid_actions, list) or any(
                not isinstance(action, str) for action in self.valid_actions
            ):
                raise LiteContractError("metadata.valid_actions must be null or a list of strings")
            self.valid_actions = list(self.valid_actions)

    @property
    def platform(self) -> Platform:
        """CUA platform derived from ``dims[0]``."""
        return self.Platform(self.dims[0])

    @property
    def task_type(self) -> TaskType:
        """CUA task type derived from ``dims[1]``."""
        return self.TaskType(self.dims[1])

    def to_dict(self) -> dict[str, Any]:
        """Serialize CUA metadata to the canonical tagged row shape."""
        return {
            "metadata_kind": self.metadata_kind,
            "dims": list(self.dims),
            "extra_tool_schemas": self.extra_tool_schemas,
            "valid_actions": self.valid_actions,
            "others": self.others,
        }

    @classmethod
    def from_dict(cls, data: Any) -> LiteCUAMetadata:
        """Build CUA metadata from a canonical tagged row."""
        if isinstance(data, cls):
            return data
        if isinstance(data, LiteBaseMetadata):
            raise LiteContractError(
                f"metadata.metadata_kind must be 'cua'; got {data.metadata_kind!r}"
            )
        if not isinstance(data, dict):
            raise LiteContractError("metadata must be a dict")
        _reject_unknown_keys(
            data,
            frozenset({
                "metadata_kind",
                "dims",
                "extra_tool_schemas",
                "valid_actions",
                "others",
            }),
        )
        _require_metadata_kind(data, cls.metadata_kind)
        return cls(
            dims=data["dims"],
            extra_tool_schemas=data.get("extra_tool_schemas", []),
            valid_actions=data.get("valid_actions"),
            others=data.get("others", {}),
        )


@dataclass(kw_only=True, slots=True)
class LiteGenericMetadata(LiteBaseMetadata):
    """Generic task metadata without CUA-only fields."""

    metadata_kind: ClassVar[str] = "generic"

    def to_dict(self) -> dict[str, Any]:
        """Serialize generic metadata to the canonical tagged row shape."""
        return {
            "metadata_kind": self.metadata_kind,
            "dims": list(self.dims),
            "extra_tool_schemas": self.extra_tool_schemas,
            "others": self.others,
        }

    @classmethod
    def from_dict(cls, data: Any) -> LiteGenericMetadata:
        """Build generic metadata from a canonical tagged row."""
        if isinstance(data, cls):
            return data
        if isinstance(data, LiteBaseMetadata):
            raise LiteContractError(
                f"metadata.metadata_kind must be 'generic'; got {data.metadata_kind!r}"
            )
        if not isinstance(data, dict):
            raise LiteContractError("metadata must be a dict")
        _reject_unknown_keys(
            data,
            frozenset({
                "metadata_kind",
                "dims",
                "extra_tool_schemas",
                "others",
            }),
        )
        _require_metadata_kind(data, cls.metadata_kind)
        return cls(
            dims=data["dims"],
            extra_tool_schemas=data.get("extra_tool_schemas", []),
            others=data.get("others", {}),
        )


def metadata_from_dict(data: Any) -> LiteBaseMetadata:
    """Build Lite task metadata from the canonical tagged row shape."""
    if isinstance(data, LiteBaseMetadata):
        return data
    if not isinstance(data, dict):
        raise LiteContractError("metadata must be a dict")
    _required_keys(data, ("metadata_kind", "dims"))
    if data["metadata_kind"] == LiteCUAMetadata.metadata_kind:
        return LiteCUAMetadata.from_dict(data)
    if data["metadata_kind"] == LiteGenericMetadata.metadata_kind:
        return LiteGenericMetadata.from_dict(data)
    raise LiteContractError(f"unknown metadata.metadata_kind {data['metadata_kind']!r}")


#: Task types where the assistant decision is the answer and no follow-up
#: observation is expected. The set is listed explicitly so adding a multiturn
#: task type is a metadata decision, not a prefix match.
SINGLE_TURN_TASK_TYPES = frozenset({
    LiteCUAMetadata.TaskType.UNDERSTANDING,
    LiteCUAMetadata.TaskType.GROUNDING_ACTION,
    LiteCUAMetadata.TaskType.GROUNDING_POINT,
    LiteCUAMetadata.TaskType.GROUNDING_BBOX,
})


__all__ = [
    "SINGLE_TURN_TASK_TYPES",
    "LiteBaseMetadata",
    "LiteCUAMetadata",
    "LiteGenericMetadata",
    "metadata_from_dict",
]
