"""Message image-reference binding helpers.

This module owns the structural rule for which Lite message roles may carry
image references. Pixel decoding, resizing, PNG/base64, and PIL helpers stay in
``lite.utils.image``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from lite.core.errors import LiteContractError
from lite.core.messages.content import IMAGE_PART, TOOL_ROLE, USER_ROLE

#: Roles whose content may carry model-bound image references. Derived from
#: the role vocabulary in ``lite.core.messages.content`` -- never re-listed.
#:
#: Same shape as its sibling :data:`lite.core.messages.turns.OBSERVATION_ROLES`
#: -- a total membership predicate with no ``is_*`` wrapper, argued once there --
#: and it happens to equal it today. The two answer different questions (may
#: carry an image part / carries a turn's observation) and must stay free to
#: diverge when a role is added.
IMAGE_REFERENCE_ROLES = frozenset({USER_ROLE, TOOL_ROLE})


@dataclass(frozen=True)
class ImageReference:
    """One image content part bound positionally from a Lite message."""

    message_index: int
    part_index: int
    role: str
    index: int


def _coerce_image_index(value: object, *, where: str = "image index") -> int:
    """Return a non-negative integer image index or raise a binding error."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise LiteContractError(f"{where} must be a non-negative integer")
    idx = int(value)
    if idx < 0:
        raise LiteContractError(f"{where} must be non-negative")
    return idx


def iter_image_references(
    messages: list[dict[str, Any]],
    *,
    validate_content_shape: bool = False,
) -> Iterator[ImageReference]:
    """Yield bindable image references in prompt/message order.

    Unsupported-role image parts raise by default so train/export and validation
    do not ignore screenshots that the model never receives.
    ``validate_content_shape`` lets canonical validators reject non-list content;
    runtime binding paths leave non-list content to their caller contracts.
    """
    for mi, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list):
            if validate_content_shape:
                raise LiteContractError(f"messages[{mi}].content must be a list or string")
            continue
        for pi, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != IMAGE_PART:
                continue
            role = msg.get("role")
            if role not in IMAGE_REFERENCE_ROLES:
                raise LiteContractError(
                    f"messages[{mi}].content[{pi}] image parts are only valid on "
                    f"{sorted(IMAGE_REFERENCE_ROLES)} messages; got role {role!r}"
                )
            idx = _coerce_image_index(
                part.get("index"),
                where=f"messages[{mi}].content[{pi}].index",
            )
            yield ImageReference(
                message_index=mi,
                part_index=pi,
                role=str(role),
                index=idx,
            )


def referenced_image_indices_in_message_order(messages: list[dict[str, Any]]) -> tuple[int, ...]:
    """Extract image indices in the same order image references appear."""
    return tuple(ref.index for ref in iter_image_references(messages))


def referenced_images_in_message_order(
    messages: list[dict[str, Any]],
    images_list: list[Any],
) -> list[Any]:
    """Collect referenced images in message image-reference order.

    Orphan images in ``images_list`` are allowed; invalid references are not.
    """
    n_images = len(images_list)
    out: list[Any] = []
    for ref in iter_image_references(messages):
        if ref.index >= n_images:
            raise LiteContractError(
                f"messages[{ref.message_index}].content[{ref.part_index}].index "
                f"{ref.index} out of range for images length {n_images}"
            )
        image = images_list[ref.index]
        if image is None:
            raise LiteContractError(
                f"messages[{ref.message_index}].content[{ref.part_index}].index "
                f"{ref.index} references an unprocessed image slot"
            )
        out.append(image)
    return out


def validate_image_references(
    messages: list[dict[str, Any]],
    images: Any,
) -> None:
    """Validate message image references without requiring every image be used.

    Only image CONTENT parts are walked. BrowserGym goal images are ordinary
    turn-0 user image parts by the time this owner validates them, so they share
    the same role and bounds rules as every other image reference.
    """
    if images is None:
        images = []
    if not isinstance(images, list):
        raise LiteContractError("images must be a list")
    n_images = len(images)
    for ref in iter_image_references(messages, validate_content_shape=True):
        if ref.index >= n_images:
            raise LiteContractError(
                f"messages[{ref.message_index}].content[{ref.part_index}].index "
                f"{ref.index} out of range for images length {n_images}"
            )


__all__ = [
    "referenced_image_indices_in_message_order",
    "referenced_images_in_message_order",
    "validate_image_references",
]
