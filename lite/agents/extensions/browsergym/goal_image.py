"""BrowserGym goal-image agent (VisualWebArena).

VWA tasks can carry task-defining goal image(s) ("find/sell THIS item") that the
model must see on every turn. The env surfaces them in the reset observation's
``metadata["goal_images_b64"]`` (a list of base64 PNGs — see ``main.py``); the
generic agent loop passes that metadata through untouched (it is env-specific,
so the loop stays general).

:class:`VisualWebArenaGoalImageAgent` is one concrete, model-agnostic agent that
adds the two browsergym-specific pieces the generic loop can't: it decodes the
goal image(s) once at turn 0 into the trajectory image arrays (only the loop owns
``processed_images``), and wraps the adapter's protocol to re-surface them
(labeled, before the per-turn screenshot) on every turn — surviving history
windowing, for any protocol (``browsergym.generic``, ``qwen3_vl.history``, …).

It is model-agnostic because it doesn't bake in a model: a config selects it by
``agent_id: "visualwebarena.goal_image"`` and points it at the model's adapter
via ``agent_kwargs.adapter_key`` (e.g. ``"qwen3_vl"`` for vision+coord,
``"qwen3_vl.base"`` for text+bid) — the same "define a concrete class, point it
at the adapter" pattern the adapters/protocols use. The same agent serves
Qwen3-VL, Qwen3.5, etc. by changing only ``adapter_key``.

A bare slug (``"qwen3_vl"``) is auto-completed with the env metadata dims
from the env metadata, exactly like ``make`` resolves ``agent_id`` — so the
config matches the rest of the codebase's convention. A fully-qualified key
(one containing ``@``, e.g. ``"qwen3_vl@browser@use"``) is used verbatim.

A model-side bridge for the browsergym env; registered via
``import lite.agents.extensions`` (pulled in by ``lite.agents.models``),
symmetric to :class:`BrowserGymGenericProtocol` in ``protocol.py``.

Run a registry import smoke by resolving
``visualwebarena.goal_image@browser@use``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import Any

from PIL import Image

from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.core.agent.base import AutoAdapterAgent
from lite.agents.types import PredictResult
from lite.core import (
    LiteMessage,
    LiteSample,
)
from lite.utils.image import decode_image
from lite.utils.registry import compose_key

logger = logging.getLogger(__name__)

# Metadata keys (env -> loop -> protocol), browsergym-internal:
#   obs.metadata["goal_images_b64"]          : list[str]  (env -> loop, raw PNGs)
#   turn-0 msg metadata["goal_image_indices"]: list[int]  (loop -> protocol)
_B64_KEY = "goal_images_b64"
_IDX_KEY = "goal_image_indices"

# Labels so the model (and anyone reading the prompt) can tell the goal
# image(s) apart from the per-turn page screenshot that follows. They are a
# RENDERING artifact only: ``_render_goal_image_labels`` splices them into
# protocol output, and nothing writes them into the persisted trajectory. A
# persisted label would become the turn-0 message's first text part and
# protocols that read the task instruction from there (notably
# ``BrowserGymGenericProtocol``'s ``## Goal:``) would render the label instead
# of the instruction.
_REF_LABEL = "Task reference image(s) for the instruction below:"
_OBS_LABEL = "Current screenshot:"


def _goal_image_indices(messages: list[LiteMessage]) -> list[int]:
    """Goal image indices recorded on the turn-0 user message's metadata, or []."""
    for m in messages:
        if m.get("role") != "user":
            continue
        for item in m.get("content", []):
            if isinstance(item, dict) and item.get("type") == "metadata":
                indices = (item.get("data") or {}).get(_IDX_KEY)
                if indices:
                    return list(indices)
        return []  # only the first (turn-0) user message carries it
    return []


def _persist_goal_images(message: LiteMessage, indices: list[int]) -> None:
    """In-place: make metadata-ordered goal images the persisted image prefix.

    Only image parts are written — the goal images are real trajectory data, so
    export/unroll paths can recover image order by walking messages. Existing
    goal image parts are moved, not merely detected, because processor image
    binding is positional: prompt image-pad order must match message image
    order. The non-goal content, including the per-turn page screenshot and the
    task instruction text, stays after the goal block.
    """
    if not indices:
        return
    content = message.get("content", [])
    content = content if isinstance(content, list) else []
    goal_set = set(indices)
    rest = [
        c for c in content
        if not (
            isinstance(c, dict)
            and c.get("type") == "image"
            and c.get("index") in goal_set
        )
    ]
    message["content"] = [
        *({"type": "image", "index": i} for i in indices),
        *rest,
    ]


def _render_goal_image_labels(messages: list[LiteMessage], indices: list[int]) -> None:
    """Add model-visible labels without changing persisted source messages."""
    if not indices:
        return
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    if first_user is None:
        return

    goal_set = set(indices)
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        message["content"] = [
            part for part in content
            if not (
                isinstance(part, dict)
                and part.get("type") == "text"
                and part.get("text") in {_REF_LABEL, _OBS_LABEL}
            )
        ]

    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        message["content"] = [
            part for part in content
            if not (
                isinstance(part, dict)
                and part.get("type") == "image"
                and part.get("index") in goal_set
            )
        ]

    first_content = first_user.get("content", [])
    if not isinstance(first_content, list):
        first_content = []
    first_user["content"] = [
        {"type": "text", "text": _REF_LABEL},
        *({"type": "image", "index": i} for i in indices),
        *first_content,
    ]

    latest: tuple[LiteMessage, int] | None = None
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for pos, part in enumerate(content):
            if (
                isinstance(part, dict)
                and part.get("type") == "image"
                and part.get("index") not in goal_set
            ):
                latest = (message, pos)
    if latest is not None:
        message, pos = latest
        message["content"].insert(pos, {"type": "text", "text": _OBS_LABEL})


def splice_goal_images(
    source_messages: list[LiteMessage],
    rendered_messages: list[LiteMessage],
) -> list[LiteMessage]:
    """Apply BrowserGym goal-image rendering to protocol-shaped messages.

    ``source_messages`` is the unwindowed trajectory prefix so turn-0 metadata
    remains visible after history protocols have dropped that turn. The splice
    mutates and returns ``rendered_messages`` for compatibility with protocol
    hooks and the live agent wrapper. It is intentionally idempotent because
    BrowserGym generic protocols and VisualWebArena wrappers can both apply it.
    """
    indices = _goal_image_indices(source_messages)
    if not indices:
        return rendered_messages
    _render_goal_image_labels(rendered_messages, indices)
    return rendered_messages


def wrap_goal_image_protocol(adapter: Any) -> None:
    """Install the VWA goal-image splice on an adapter's protocol.

    The splice runs after the adapter protocol has applied history windowing, so
    the reference images survive even when turn 0 has fallen out of the normal
    image budget. This is shared by live rollout and offline export.
    """
    protocol = adapter.protocol
    if getattr(protocol, "_cua_lite_goal_image_wrapped", False):
        return
    base_process_messages = protocol.process_messages

    def process_messages(messages: list[LiteMessage], **kwargs) -> list[LiteMessage]:
        return splice_goal_images(messages, base_process_messages(messages, **kwargs))

    protocol.process_messages = process_messages
    protocol._cua_lite_goal_image_wrapped = True


@dataclass
class VisualWebArenaGoalImageAgent(
    AutoAdapterAgent, key=r"visualwebarena\.goal_image(@browser@use)?"
):
    """Concrete model-agnostic agent that shows the env's goal image(s) every
    turn. Selected via ``agent_id: "visualwebarena.goal_image"``; pointed at the
    model adapter via ``agent_kwargs.adapter_key``."""

    def __post_init__(self) -> None:
        # Point at the model's adapter (this agent is model-agnostic, so it does
        # NOT resolve the adapter from its own key like AutoAdapterAgent does).
        adapter_kwargs = self._adapter_kwargs()
        adapter_key = adapter_kwargs.pop("adapter_key", None)
        if not adapter_key:
            raise ValueError(
                "goal_image agent requires agent_kwargs.adapter_key — the model "
                "adapter to point at, e.g. 'qwen3_vl' (vision+coord) or "
                "'qwen3_vl.base' (text+bid)."
            )
        # Env-derived @dims completion, mirroring make's
        # agent-key rule (lite/agents/factory.py): a bare model slug like
        # "qwen3_vl" is completed from the
        # env metadata (forwarded into self.kwargs by make). A key that
        # already names a fully-qualified key (contains "@") is used verbatim,
        # so existing fully-qualified configs keep working. Stays
        # model-agnostic — only the @dims suffix is env-derived; the model is
        # the caller's
        # choice. (".base" keys get the suffix appended too, e.g.
        # "qwen3_vl.base@browser@use", which the base adapter's
        # platform/task pattern absorbs → the same base adapter.)
        if "@" not in adapter_key:
            meta = adapter_kwargs.get("metadata")
            if meta is not None:
                adapter_key = compose_key(adapter_key, *meta.dims)
        self.adapter = AgentAdapterRegistry.get(adapter_key, **adapter_kwargs)

        wrap_goal_image_protocol(self.adapter)

    def build_generation_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Forward ``enable_thinking`` to the chat template when the adapter has
        it (Qwen3-VL/Qwen3.5 thinking variants); model-agnostic otherwise."""
        if self.processor is None:
            raise RuntimeError("agent.processor is not set")
        extra: dict[str, Any] = {}
        if hasattr(self.adapter, "enable_thinking"):
            extra["enable_thinking"] = self.adapter.enable_thinking
        return self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **extra,
        )

    async def _ingest_goal_images(
        self, lite_sample: LiteSample, processed_images: list[Image.Image]
    ) -> None:
        """Decode ``metadata["goal_images_b64"]`` from the turn-0 user message
        into ordinary turn-0 image parts, recording their indices back on that
        metadata item as ``goal_image_indices`` (the carrier the protocol splice
        reads). Idempotent: the raw base64 is removed once decoded."""
        for msg in lite_sample.messages:
            if msg.get("role") != "user":
                continue
            for item in msg.get("content", []):
                if item.get("type") != "metadata":
                    continue
                data = item.get("data") or {}
                if _B64_KEY not in data:
                    continue
                indices: list[int] = []
                for b64 in data[_B64_KEY]:
                    img = await asyncio.to_thread(
                        decode_image, base64.b64decode(b64)
                    )
                    lite_sample.images.append(img)
                    processed_images.append(
                        await asyncio.to_thread(self.adapter.process_image, img)
                    )
                    indices.append(len(lite_sample.images) - 1)
                data[_IDX_KEY] = indices
                del data[_B64_KEY]
                _persist_goal_images(msg, indices)
                logger.info("ingested %d goal image(s) at %s", len(indices), indices)
            return  # only the first (turn-0) user message carries reset metadata

    async def _predict_with_details(
        self,
        lite_sample: LiteSample,
        *,
        processed_images: list[Image.Image],
        call_id_start: int = 0,
    ) -> PredictResult:
        await self._ingest_goal_images(lite_sample, processed_images)
        return await super()._predict_with_details(
            lite_sample,
            processed_images=processed_images,
            call_id_start=call_id_start,
        )
