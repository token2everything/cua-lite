"""Qwen2.5-VL agent — one-step creation via AgentRegistry.

The agent subclass forwards the adapter's ``enable_thinking`` flag to
``processor.apply_chat_template`` (which Qwen2.5-VL's chat_template silently
ignores; the forward keeps interface parity with the Qwen3-VL agent path).

Usage::

    agent = AgentRegistry.get(
        "qwen2_5_vl@desktop@grounding.point",
        processor=processor,
        generate_fn=generate_fn,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lite.agents.core.agent import AutoAdapterAgent


@dataclass
class Qwen2_5VLBaseAgent(
    AutoAdapterAgent,
    key=(
        r"qwen2_5_vl\.base"
        r"(@(desktop|browser|mobile)"
        r"@(use|understanding|grounding\.action|grounding\.point|grounding\.bbox))?"
    ),
):
    """Base agent class for the Qwen2.5-VL family.

    Mirrors :class:`Qwen3VLBaseAgent` — forwards ``adapter.enable_thinking``
    to ``apply_chat_template`` so the interface stays uniform. Qwen2.5-VL's
    chat_template does not consume the flag (no native ``<think>`` channel),
    but the forward is harmless.
    """

    def build_generation_prompt(self, messages: list[dict[str, Any]]) -> str:
        if self.processor is None:
            raise RuntimeError("agent.processor is not set")
        enable_thinking = getattr(self.adapter, "enable_thinking", False)
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )


# Concrete agent subclasses, one per (platform, task_type). Same regex pattern
# as the adapter package so registry resolution is symmetric.

@dataclass
class Qwen2_5VLDesktopGroundingPointAgent(
    Qwen2_5VLBaseAgent, key=r"qwen2_5_vl@(desktop|browser)@grounding\.point"
):
    """Desktop/browser point-grounding registry entry."""
    pass


@dataclass
class Qwen2_5VLDesktopGroundingActionAgent(
    Qwen2_5VLBaseAgent, key=r"qwen2_5_vl@(desktop|browser)@grounding\.action"
):
    """Desktop/browser action-grounding registry entry."""
    pass


@dataclass
class Qwen2_5VLMobileGroundingPointAgent(
    Qwen2_5VLBaseAgent, key="qwen2_5_vl@mobile@grounding.point"
):
    """Mobile point-grounding registry entry."""
    pass


@dataclass
class Qwen2_5VLMobileGroundingActionAgent(
    Qwen2_5VLBaseAgent, key="qwen2_5_vl@mobile@grounding.action"
):
    """Mobile action-grounding registry entry."""
    pass


@dataclass
class Qwen2_5VLDesktopUseAgent(
    Qwen2_5VLBaseAgent, key=r"qwen2_5_vl@(desktop|browser)@use"
):
    """Desktop/browser GUI-use registry entry."""
    pass


@dataclass
class Qwen2_5VLMobileUseAgent(
    Qwen2_5VLBaseAgent, key="qwen2_5_vl@mobile@use"
):
    """Mobile GUI-use registry entry."""
    pass
