"""UITars 1.5 v1 agent — one-step creation via AgentRegistry.

Usage:
    agent = AgentRegistry.get("ui_tars_15_v1@desktop@use", processor=processor, generate_fn=fn)
    agent = AgentRegistry.get("ui_tars_15_v1@browser@use", processor=processor, generate_fn=fn)
    agent = AgentRegistry.get("ui_tars_15_v1@mobile@use", processor=processor, generate_fn=fn)
"""

from __future__ import annotations

from dataclasses import dataclass

from lite.agents.core.agent import AutoAdapterAgent

# Desktop and browser share one agent class per task type via the
# ``(desktop|browser)`` regex key. Mobile is its own platform.

@dataclass
class UITars15V1DesktopUseAgent(AutoAdapterAgent, key=r"ui_tars_15_v1@(desktop|browser)@use"):
    """Desktop/browser GUI-use registry entry."""
    pass

@dataclass
class UITars15V1DesktopGroundingActionAgent(AutoAdapterAgent, key=r"ui_tars_15_v1@(desktop|browser)@grounding\.action"):
    """Desktop/browser action-grounding registry entry."""
    pass

@dataclass
class UITars15V1DesktopGroundingPointAgent(AutoAdapterAgent, key=r"ui_tars_15_v1@(desktop|browser)@grounding\.point"):
    """Desktop/browser point-grounding registry entry."""
    pass

@dataclass
class UITars15V1MobileUseAgent(AutoAdapterAgent, key="ui_tars_15_v1@mobile@use"):
    """Mobile GUI-use registry entry."""
    pass

@dataclass
class UITars15V1MobileGroundingActionAgent(AutoAdapterAgent, key="ui_tars_15_v1@mobile@grounding.action"):
    """Mobile action-grounding registry entry."""
    pass

@dataclass
class UITars15V1MobileGroundingPointAgent(AutoAdapterAgent, key="ui_tars_15_v1@mobile@grounding.point"):
    """Mobile point-grounding registry entry."""
    pass
