"""Step-GUI (GELab-Zero) mobile agent — one-step creation via AgentRegistry.

The wire-format fold (cua-lite ``LiteMessage`` →
``<THINK>...</THINK>\\nexplain:...\\taction:...\\t...`` SFT text) lives in
:meth:`STEPGUIMobileBaseAdapter.convert_message_to_agent`, so the agent itself
needs no override.

Usage::

    agent = AgentRegistry.get(
        "step_gui@mobile@use",
        processor=processor,
        generate_fn=generate_fn,
        protocol_kwargs={"full_history_size": 3},
    )
"""

from __future__ import annotations

from dataclasses import dataclass

from lite.agents.core.agent import AutoAdapterAgent


@dataclass
class STEPGUIMobileAgent(AutoAdapterAgent, key="step_gui@mobile@use"):
    """Mobile GUI-use registry entry."""
    pass


__all__ = ["STEPGUIMobileAgent"]
