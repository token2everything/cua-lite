"""Fara-1.0 agent — one-step creation via AgentRegistry.

Mirrors :class:`Qwen2_5VLBaseAgent` — forwards ``adapter.enable_thinking`` to
``apply_chat_template`` for interface parity (Fara / Qwen2.5-VL have no native
``<think>`` channel, so the flag is a harmless no-op).

Usage::

    agent = AgentRegistry.get(
        "fara@browser@use",
        processor=processor,
        generate_fn=generate_fn,
    )
"""

from __future__ import annotations

from dataclasses import dataclass

from lite.agents.models.qwen2_5_vl.agent import Qwen2_5VLBaseAgent


@dataclass
class FaraBaseAgent(
    Qwen2_5VLBaseAgent,
    key=r"fara\.base(@(desktop|browser)@(use|grounding\.action|grounding\.point))?",
):
    """Base agent class for the Fara-1.0 family."""
    pass


@dataclass
class FaraDesktopGroundingActionAgent(
    FaraBaseAgent, key=r"fara@(desktop|browser)@grounding\.action"
):
    """Desktop/browser action-grounding registry entry."""
    pass


@dataclass
class FaraDesktopGroundingPointAgent(
    FaraBaseAgent, key=r"fara@(desktop|browser)@grounding\.point"
):
    """Desktop/browser point-grounding registry entry."""
    pass


@dataclass
class FaraDesktopUseAgent(
    FaraBaseAgent, key=r"fara@(desktop|browser)@use"
):
    """Desktop/browser GUI-use registry entry."""
    pass
