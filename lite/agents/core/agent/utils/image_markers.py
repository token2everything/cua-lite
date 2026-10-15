"""Local-only image index markers for provider request assembly.

Mark/read/strip mechanics only. Provider families own their own image-block
traversal: this module never inspects provider block shapes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

PROVIDER_VISIBLE_IMAGE_INDEX_MARKER = "_cua_lite_image_index"


def mark_provider_visible_image_index(
    block: dict[str, Any],
    image_index: int | None,
) -> dict[str, Any]:
    """Attach the local-only image index marker to a provider-visible image block."""
    if image_index is None:
        return block
    marked = dict(block)
    marked[PROVIDER_VISIBLE_IMAGE_INDEX_MARKER] = image_index
    return marked


def provider_visible_image_indices_from_marked_blocks(blocks: Iterable[Any]) -> tuple[int, ...]:
    """Collect image indices from the provider-selected image blocks."""
    image_indices: list[int] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        image_index = block.get(PROVIDER_VISIBLE_IMAGE_INDEX_MARKER)
        if isinstance(image_index, int):
            image_indices.append(image_index)
    return tuple(image_indices)


def strip_provider_visible_image_index_marker(block: dict[str, Any]) -> dict[str, Any]:
    """Remove the local image index marker from one provider image block."""
    if PROVIDER_VISIBLE_IMAGE_INDEX_MARKER not in block:
        return block
    return {
        key: value for key, value in block.items() if key != PROVIDER_VISIBLE_IMAGE_INDEX_MARKER
    }


__all__ = [
    "mark_provider_visible_image_index",
    "provider_visible_image_indices_from_marked_blocks",
    "strip_provider_visible_image_index_marker",
]
