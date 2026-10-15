"""EvoCUA agent — one-step creation via AgentRegistry.

Inherits :class:`Qwen3VLBaseAgent` (EvoCUA's adapter family extends
:class:`Qwen3VLBaseAdapter`) so the ``adapter.enable_thinking`` flag is
forwarded to ``processor.apply_chat_template(..., enable_thinking=...)``
— without this, the inherited ``enable_thinking`` field would be
silently no-op'd on EvoCUA.

Usage:
    agent = AgentRegistry.get(
        "evocua@desktop@use",
        processor=processor,
        generate_fn=generate_fn,
        protocol_kwargs={"full_history_size": 4},
    )
"""

from __future__ import annotations

from dataclasses import dataclass

from lite.agents.models.qwen3_vl.agent import Qwen3VLBaseAgent

# Desktop and browser share one agent class per task type via the
# ``(desktop|browser)`` regex key. EvoCUA doesn't ship a mobile variant.

@dataclass
class EvoCUADesktopUseAgent(Qwen3VLBaseAgent, key=r"evocua@(desktop|browser)@use"):
    """Desktop/browser GUI-use registry entry."""
    pass

@dataclass
class EvoCUADesktopGroundingActionAgent(Qwen3VLBaseAgent, key=r"evocua@(desktop|browser)@grounding\.action"):
    """Desktop/browser action-grounding registry entry."""
    pass

@dataclass
class EvoCUADesktopGroundingPointAgent(Qwen3VLBaseAgent, key=r"evocua@(desktop|browser)@grounding\.point"):
    """Desktop/browser point-grounding registry entry."""
    pass
