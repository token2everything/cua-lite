"""UITars agent (original UI-TARS-7B-DPO) — one-step creation via AgentRegistry.

Usage:
    agent = AgentRegistry.get("ui_tars@desktop@use", processor=processor, generate_fn=fn)
    agent = AgentRegistry.get("ui_tars@browser@use", processor=processor, generate_fn=fn)
    agent = AgentRegistry.get("ui_tars@mobile@use", processor=processor, generate_fn=fn)
"""

from __future__ import annotations

from dataclasses import dataclass

from lite.agents.core.agent import AutoAdapterAgent

# Desktop and browser share one agent class per task type via the
# ``(desktop|browser)`` regex key. Mobile is its own platform.

@dataclass
class UITarsDesktopUseAgent(AutoAdapterAgent, key=r"ui_tars@(desktop|browser)@use"):
    """Desktop/browser GUI-use registry entry."""
    pass

@dataclass
class UITarsDesktopGroundingActionAgent(AutoAdapterAgent, key=r"ui_tars@(desktop|browser)@grounding\.action"):
    """Desktop/browser action-grounding registry entry."""
    pass

@dataclass
class UITarsDesktopGroundingPointAgent(AutoAdapterAgent, key=r"ui_tars@(desktop|browser)@grounding\.point"):
    """Desktop/browser point-grounding registry entry."""
    pass

@dataclass
class UITarsMobileUseAgent(AutoAdapterAgent, key="ui_tars@mobile@use"):
    """Mobile GUI-use registry entry."""
    pass

@dataclass
class UITarsMobileGroundingActionAgent(AutoAdapterAgent, key="ui_tars@mobile@grounding.action"):
    """Mobile action-grounding registry entry."""
    pass

@dataclass
class UITarsMobileGroundingPointAgent(AutoAdapterAgent, key="ui_tars@mobile@grounding.point"):
    """Mobile point-grounding registry entry."""
    pass
