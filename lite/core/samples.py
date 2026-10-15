"""Canonical Lite trajectory and rollout sample containers."""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from lite.core.messages.content import (
    LiteMessage,
    validate_message_content_parts,
    validate_message_roles,
)
from lite.core.messages.final import PARSE_FAILURE_FINAL_REASON
from lite.core.messages.image_refs import validate_image_references
from lite.core.metadata import LiteBaseMetadata, LiteCUAMetadata, metadata_from_dict


@dataclass
class LiteSample:
    """Trajectory representation in CUA-Lite.

    Messages reference images by index placeholders; the actual PIL images are
    stored separately in ``images``.

    ``metadata`` is always a :class:`LiteBaseMetadata`. Row metadata is normalized
    once at :meth:`from_dict`; a live sample never carries a raw dict or a JSON
    string as a second metadata shape.
    """

    metadata: LiteBaseMetadata
    images: list[Image.Image] = field(default_factory=list)
    messages: list[LiteMessage] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize the durable trajectory fields.

        The durable shape is exactly ``metadata``, ``images``, and ``messages``.
        ``images`` stay live PIL objects here; encoding them is caller-owned.
        """
        return {
            "metadata": self.metadata.to_dict(),
            "images": self.images,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data) -> LiteSample:
        """Build a trajectory from a deserialized row.

        This is the only metadata normalization point: ``metadata`` must be a
        canonical row dict (or an already-built ``LiteBaseMetadata``), and
        :func:`metadata_from_dict` owns its legal values. Role, content, and
        image-reference contracts are validated here before returning a
        ``LiteSample``.
        """
        if isinstance(data, cls):
            return data
        meta = metadata_from_dict(data.get("metadata", {}))
        messages = data.get("messages", [])
        images = data.get("images", [])
        validate_message_roles(messages)
        validate_message_content_parts(messages)
        validate_image_references(messages, images)
        return cls(
            images=images,
            messages=messages,
            metadata=meta,
        )


# Status values emitted on each LiteRLStep. They mirror slime's Sample.Status
# strings ('completed' / 'truncated' / 'aborted' / 'failed') so train-time
# conversion stays a string lookup at the framework boundary.
STATUS_COMPLETED = "completed"
STATUS_TRUNCATED = "truncated"
STATUS_ABORTED = "aborted"
STATUS_FAILED = "failed"

#: Every legal ``LiteRLStep.status``, ordered least to most severe outcome. This
#: is the only declaration of the vocabulary: a reader that rejects an
#: out-of-vocabulary status, or that ranks the worst status over a group of
#: turns, reads this tuple instead of respelling the four names. The order is
#: load-bearing for the ranking readers.
STEP_STATUSES_BY_SEVERITY = (
    STATUS_COMPLETED,
    STATUS_TRUNCATED,
    STATUS_ABORTED,
    STATUS_FAILED,
)

# Stop-reason PUBLICATION policy. The vocabulary itself is owned by
# ``lite/core/messages/final.py``; deciding which of its values survive into the
# durable record is a property of the durable record, so it lives here with the
# containers that carry it -- and deliberately NOT in the core message API
# (guarded by ``tests/core/messages/test_final_turn.py``).

#: The only stop reasons allowed into durable/public rollout metadata. Routine
#: ``content_only_final`` / ``empty`` finals stay internal runtime labels on the
#: terminal step result; everything else is dropped rather than published.
PERSISTED_FINAL_STOP_REASONS = frozenset({PARSE_FAILURE_FINAL_REASON})


@dataclass
class LiteRLStep:
    """One durable turn of rollout/train state.

    Field groups: (1) inputs the model saw at this step, (2) outputs the model
    emitted, (3) post-step env outcome.

    SFT parquet serialization is owned by
    ``lite.train.export.sft_tokenize.serialize_rl_step``.
    """

    # (1) inputs
    prompt: str
    image_indices: tuple[int, ...]

    # (2) outputs
    response: str
    response_tokens: list[int] = field(default_factory=list)
    response_log_probs: list[float] = field(default_factory=list)

    # (3) outcome
    reward: float = 0.0
    status: str = STATUS_COMPLETED

    # Precomputed prompt token ids. Populated only for text-only steps at SFT
    # export; image steps require processor-specific vision-token expansion.
    prompt_tokens: list[int] | None = None


@dataclass
class LiteRLSample:
    """Durable rollout trajectory shared by logging and train rollouts.

    Images live once on ``processed_images``; per-turn state lives in ``steps``
    records that reference images by index. Slime transport identity
    (``group_index``, ``index``, ``label``) stays on emitted slime ``Sample``
    objects rather than this framework-neutral shape.

    **No ``to_dict``/``from_dict``, deliberately.** This object is transport,
    not a durable record: it is built in memory, consumed by the agent hooks and
    the rollout segmenter, and never itself serialized `[measured: zero
    ``asdict``/``__dict__``/``json.dumps`` sites]`. What IS persisted is its
    ``lite_sample`` (via :meth:`LiteSample.to_dict`) and its ``steps`` (via the
    exporter's step projection). A pair here would have no consumer.
    """

    # ``lite_sample`` is the adapter-agnostic golden record (metadata + raw
    # images + structured messages). ``processed_images`` is index-aligned with
    # ``lite_sample.images``: referenced entries hold the student-resolution
    # form used for RL tokenization; unreferenced stored frames may stay ``None``.
    lite_sample: LiteSample = field(
        default_factory=lambda: LiteSample(metadata=LiteCUAMetadata())
    )
    processed_images: list[Image.Image | None] = field(default_factory=list)
    steps: list[LiteRLStep] = field(default_factory=list)

    episode_return: float = 0.0
    terminated: bool = False
    truncated: bool = False


__all__ = [
    "STATUS_COMPLETED",
    "STATUS_TRUNCATED",
    "STATUS_ABORTED",
    "STATUS_FAILED",
    "STEP_STATUSES_BY_SEVERITY",
    "PERSISTED_FINAL_STOP_REASONS",
    "LiteSample",
    "LiteRLStep",
    "LiteRLSample",
]
