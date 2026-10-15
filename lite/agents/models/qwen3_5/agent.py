"""Qwen3.5 agent — one-step creation via AgentRegistry.

Inherits :class:`Qwen3VLBaseAgent`'s ``build_generation_prompt`` (which
forwards ``adapter.enable_thinking`` to
``processor.apply_chat_template(..., enable_thinking=...)``). Qwen3.5's
chat_template defaults ``<think>`` to ON (prepending ``<think>\\n`` to
the generation), which (a) inflates token count with multi-hundred-token
reasoning and (b) can hit ``max_new_tokens`` before the model finishes
the ``<tool_call>`` block.

Usage::

    agent = AgentRegistry.get(
        "qwen3_5@desktop@use",
        processor=processor,
        generate_fn=generate_fn,
        # Override adapter kwargs if you want native <think>:
        adapter_kwargs={"enable_thinking": True},
        # Or leave default — thinking is OFF.
        protocol_kwargs={"history_n": 100},
    )
"""

from __future__ import annotations

from dataclasses import dataclass

from lite.agents.models.qwen3_vl.agent import Qwen3VLBaseAgent


@dataclass
class Qwen3_5BaseAgent(
    Qwen3VLBaseAgent,
    key=(
        r"qwen3_5\.base"
        r"(@(desktop|browser|mobile)"
        r"@(use|understanding|grounding\.action|grounding\.point|grounding\.bbox))?"
    ),
):
    """Base class for Qwen3.5 agents. Inherits ``build_generation_prompt``
    from :class:`Qwen3VLBaseAgent` so the ``adapter.enable_thinking``
    forwarding is shared with the Qwen3-VL family.

    Registered under a platform/task wildcard key to
    pair with :class:`Qwen3_5BaseAdapter`'s wildcard registration —
    yaml-driven rollouts that set ``agent_id: "qwen3_5.base"`` resolve
    here (workflow-agnostic, no Action:/Thought: machinery). Concrete
    subclasses below carry platform/task-specific keys
    (``qwen3_5@desktop@use`` etc.) for the ``use`` flavor.
    """


# Desktop and browser share one agent class per task type — the
# (desktop|browser) regex makes registry lookups under either platform resolve
# to the same class. Browser nav is an env extra_tool, not a per-platform
# agent. Mobile is its own platform with different action space + protocol.

@dataclass
class Qwen3_5DesktopUseAgent(
    Qwen3_5BaseAgent, key=r"qwen3_5@(desktop|browser)@use"
):
    """Desktop/browser GUI-use registry entry."""
    pass


@dataclass
class Qwen3_5DesktopGroundingActionAgent(
    Qwen3_5BaseAgent, key=r"qwen3_5@(desktop|browser)@grounding\.action"
):
    """Desktop/browser action-grounding registry entry."""
    pass


@dataclass
class Qwen3_5DesktopGroundingPointAgent(
    Qwen3_5BaseAgent, key=r"qwen3_5@(desktop|browser)@grounding\.point"
):
    """Desktop/browser point-grounding registry entry."""
    pass


@dataclass
class Qwen3_5MobileUseAgent(Qwen3_5BaseAgent, key="qwen3_5@mobile@use"):
    """Mobile GUI-use registry entry."""
    pass


@dataclass
class Qwen3_5MobileGroundingActionAgent(Qwen3_5BaseAgent, key="qwen3_5@mobile@grounding.action"):
    """Mobile action-grounding registry entry."""
    pass


@dataclass
class Qwen3_5MobileGroundingPointAgent(Qwen3_5BaseAgent, key="qwen3_5@mobile@grounding.point"):
    """Mobile point-grounding registry entry."""
    pass
