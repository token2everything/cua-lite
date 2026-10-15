"""Agent/provider provenance helpers for merged canonical action-batch calls."""

from __future__ import annotations

from dataclasses import dataclass

from lite.core.tools.action_space.batches import LiteActionBatchMergeResult
from lite.core.tools.calls import tool_call_id


@dataclass(frozen=True)
class ProviderCallProvenance:
    """Canonical Lite provenance for one provider tool call."""

    canonical_call_id: str | None
    is_final_for_canonical: bool


def provider_call_provenance_from_merge(
    provider_source_groups: list[list[int] | None],
    merge: LiteActionBatchMergeResult,
) -> tuple[ProviderCallProvenance, ...]:
    """Map provider calls to canonical ids and action-batch finality.

    ``None`` source groups are intentionally unparsed provider calls. Non-empty
    groups point into the parser's draft calls; merge provenance maps the final
    draft call in each group to the stamped canonical call it produced.
    """
    call_ids: list[str | None] = []
    for source_group in provider_source_groups:
        if source_group is None:
            call_ids.append(None)
            continue
        merged_idx = merge.source_to_merged[source_group[-1]]
        call_id = tool_call_id(merge.tool_calls[merged_idx])
        if call_id is None:
            raise ValueError(
                f"provider source group maps to merged call {merged_idx}, "
                "but that call has no canonical id"
            )
        call_ids.append(call_id)
    return tuple(
        ProviderCallProvenance(
            canonical_call_id=call_id,
            is_final_for_canonical=call_id is not None and call_id not in call_ids[idx + 1 :],
        )
        for idx, call_id in enumerate(call_ids)
    )


__all__ = [
    "ProviderCallProvenance",
    "provider_call_provenance_from_merge",
]
