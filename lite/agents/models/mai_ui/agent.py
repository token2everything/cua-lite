"""MAI-UI mobile agent — one-step creation via AgentRegistry.

The wire-format fold (cua-lite ``LiteMessage`` →
``<thinking>...</thinking>\\n<tool_call>{json}</tool_call>`` SFT text) lives in
:meth:`MAIUIMobileUseAdapter.convert_message_to_agent`, so the agent itself
needs no override.

Usage::

    agent = AgentRegistry.get(
        "mai_ui@mobile@use",
        processor=processor,
        generate_fn=generate_fn,
        protocol_kwargs={"full_history_size": 3},
    )
"""

from __future__ import annotations

from dataclasses import dataclass

from lite.agents.core.agent import AutoAdapterAgent


@dataclass
class MAIUIMobileAgent(AutoAdapterAgent, key="mai_ui@mobile@use"):
    """Mobile GUI-use registry entry."""
    pass


# MAI-UI grounding ships one harness shared across all three platforms (the
# upstream cookbook applies the same prompt to ScreenSpot-Pro / OSWorld-G /
# mobile benchmarks).

@dataclass
class MAIUIGroundingPointAgent(
    AutoAdapterAgent, key=r"mai_ui@(desktop|browser|mobile)@grounding\.point"
):
    """Desktop/browser/mobile point-grounding registry entry."""
    pass


__all__ = [
    "MAIUIMobileAgent",
    "MAIUIGroundingPointAgent",
]
