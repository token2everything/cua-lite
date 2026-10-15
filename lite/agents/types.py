"""The agent-tier rendered-trajectory and predict-loop types.

Owns exactly what the agent tier adds on top of the shared protocol:
:data:`AgentMessage` / :data:`AgentStep` / :class:`AgentSample` (an adapter's
rendered trajectory) and :data:`GenerateFn` / :class:`PredictResult` (the
sampling loop's per-turn output). Shared protocol types are owned by
``lite.core`` and are imported from there at the use site — this module is no
longer a re-export of them.

The two ``lite.core`` names below are ORDINARY imports, not re-exports: they
annotate :class:`PredictResult`.

Run:
    uv run pytest tests/agents -q
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from lite.core import LiteMessage, LiteRLStep

# Adapter output type. Plain dict shape (only ``text`` / ``image`` content
# parts) for chat_template / model API consumption — does NOT carry the rich
# LiteMessage content types like ``inline_reasoning`` / ``action_description``.
# The rich-types layer ends at ``convert_message_to_agent``; everything past
# that point is plain.
AgentMessage = dict[str, Any]


# One turn's rendered messages. Image references inside survive as native
# ImageContent = {"type": "image", "index": int} content parts pointing into
# the enclosing AgentSample.processed_images.
AgentStep = list[AgentMessage]


@dataclass
class AgentSample:
    """Adapter's rendered trajectory.

    One row in the SFT parquet v2 corresponds to one of these. Images live
    once on the trajectory; each turn's view is an :data:`AgentStep` whose
    ImageContent.index entries point into ``processed_images``.

    Attributes:
        processed_images: Trajectory-level image list, index-aligned with the
            raw Lite trajectory images. Entries referenced by a step are the
            adapter-processed images; unreferenced stored frames may be ``None``.
        steps: One :data:`AgentStep` per turn (= one rendered list of
            ``AgentMessage``). The first ``AgentStep`` always begins with
            the system prompt + first user observation; subsequent steps
            extend the conversation.
        metadata: Pass-through ``LiteSample.metadata`` (task_type, platform,
            extra_tool_schemas, valid_actions, others.*). Used by the SFT
            collator and downstream filters.
    """
    processed_images: list[Image.Image | None] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__}


# =============================================================================
# Runtime data types (sampling loop)
# =============================================================================

# Type alias for the VLM generate function:
#   async (**kwargs) -> {"response": str, ...}
# Called with: prompt, images, messages. Must return a dict with required
# "response" key. Other keys (response_tokens, response_log_probs,
# finish_reason, ...) are pulled into ``LiteRLStep`` explicitly by the
# agent's sample loop.
GenerateFn = Callable[..., Awaitable[dict[str, Any]]]


@dataclass
class PredictResult:
    """Full predict output including intermediate data for logging.

    Contains a ``step`` (the per-turn :class:`LiteRLStep` ready for RL
    training) plus the parsed Lite message and original agent message
    (the latter mostly for trajectory-logger sidecars).
    """
    lite_message: LiteMessage
    agent_message: AgentMessage
    step: LiteRLStep
    model_output_error: str | None = None


__all__ = [
    "AgentMessage",
    "AgentStep",
    "AgentSample",
    "GenerateFn",
    "PredictResult",
]
