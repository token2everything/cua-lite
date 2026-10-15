"""
Qwen3.8 Adapters (Qwen3.5 XML wire format, expanded desktop action enum).

Qwen3.8 is built on the Qwen3.5 architecture (``config.json`` reports
``model_type: "qwen3_5"``) and ships the same ``chat_template.jinja`` tool
grammar — ``<tool_call><function=NAME><parameter=K>…</parameter></function></tool_call>``.
So the whole XML rendering/parsing stack is inherited from
:mod:`lite.agents.models.qwen3_5.adapter` unchanged.

What this module owns is the delta introduced by the expanded OSWorld harness
(``${CUA_LITE_REFERENCES_ROOT}/OSWorld/mm_agents/qwen/``, exported ``QwenAgent``):

* :data:`QWEN38_USE_SYSTEM_PROMPT` — the expanded ``# Response format``
  section, which additionally licenses a *terminal turn with no tool call*
  (``prompts.build_internal_system_prompt``). The Qwen3.5 prompt demands a
  ``<tool_call>`` on every turn.
* desktop/browser adapters bind :class:`Qwen3_8DesktopActionSpace` (19 action
  values, ``call_user`` instead of ``answer``).

The no-tool-call turn needs no parser change: ``parse_raw_assistant_response``
already leaves ``tool_calls`` unset and keeps the prose as a text part, and
``Qwen3_5UseAdapter.convert_message_from_agent`` already routes a
tool-call-less turn down the text-final path untouched.

Mobile keeps the Qwen3.5 surface (``mobile_use`` + :data:`USE_SYSTEM_PROMPT`):
the expanded harness declares no mobile tool, so there is no upstream delta to
project.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.adapter import AgentAdapterRegistry, AsIsAdapter
from lite.agents.core.protocol.base import FullHistoryProtocol
from lite.agents.models.qwen3_5.adapter import Qwen3_5BaseAdapter, Qwen3_5UseAdapter
from lite.agents.models.qwen3_5.protocol import Qwen3_5HistoryProtocol
from lite.agents.models.qwen3_8.action_space import (
    Qwen3_8DesktopActionSpace,
    Qwen3_8DesktopGroundingPointActionSpace,
    Qwen3_8MobileActionSpace,
    Qwen3_8MobileGroundingPointActionSpace,
)
from lite.agents.models.qwen3_vl.adapter import GROUNDING_POINT_SYSTEM_PROMPT
from lite.core import LiteCUAMetadata
from lite.core.tools.extra_tools import LiteAppLaunchToolSet, LiteFinishToolSet

# =============================================================================
# System Prompts
# =============================================================================

# Expanded ``use`` response format. Mirrors the upstream
# ``build_internal_system_prompt`` tail. Two differences from the Qwen3.5
# prompt matter at runtime: a terminal turn may carry prose with no tool call,
# and the terminal tool calls are ``call_user`` / ``terminate`` (the expanded
# enum has no ``answer``).
#
# The infeasibility line is upstream's wording. cua-lite scores refusal through
# the env-gated ``report_infeasible`` extra tool when a row exposes it; this
# line only tells the model to be explicit in prose, which costs nothing when
# the tool is absent.
QWEN38_USE_SYSTEM_PROMPT = """# Response format

For normal UI interaction steps:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block.

For terminal steps, you may either:
- output a final natural-language response with no tool call, or
- use a terminal tool call such as call_user or terminate.

Rules:
- For non-terminal UI steps, output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything after a tool call.
- Use call_user when you need user information or confirmation.
- Use terminate when you want to explicitly end the task with a success or failure status.
- If the task is infeasible, say so explicitly in the response."""


# =============================================================================
# Base adapter
# =============================================================================


@dataclasses.dataclass
class Qwen3_8BaseAdapter(
    Qwen3_5BaseAdapter,
    key=(
        r"qwen3_8\.base"
        r"(@(desktop|browser|mobile)"
        r"@(use|understanding|grounding\.action|grounding\.point|grounding\.bbox))?"
    ),
):
    """Workflow-agnostic Qwen3.8 adapter.

    Adds nothing to :class:`Qwen3_5BaseAdapter` but the registry key: the XML
    tools section, inline XML tool_call rendering, message order, and
    ``enable_thinking`` forwarding are all identical between the two releases.
    yaml-driven rollouts that set ``agent_id: "qwen3_8.base"`` resolve here.
    """


# =============================================================================
# Desktop + Browser Adapters
# =============================================================================


# Desktop and browser share one adapter class per task type, same as Qwen3.5:
# browser nav verbs arrive as env extra_tools through the shared action space.


@dataclasses.dataclass
class Qwen3_8DesktopGroundingActionAdapter(
    Qwen3_8BaseAdapter, key=r"qwen3_8@(desktop|browser)@grounding\.action"
):
    """Desktop+browser grounding/action: full expanded action vocabulary, single turn.

    Not routed by env eval (envs declare ``grounding.point``); kept for SFT
    replay of family-native XML with the expanded desktop enum.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen3_8DesktopActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    natively_rendered_extra_tool_names: ClassVar[frozenset[str]] = (
        LiteFinishToolSet.get_tool_names()
    )


@dataclasses.dataclass
class Qwen3_8DesktopGroundingPointAdapter(
    Qwen3_8BaseAdapter, key=r"qwen3_8@(desktop|browser)@grounding\.point"
):
    """Desktop+browser grounding (single-step click) for Qwen3.8.

    Uses the trimmed :class:`Qwen3_8DesktopGroundingPointActionSpace` plus the
    minimal :data:`GROUNDING_POINT_SYSTEM_PROMPT` — a one-click surface is
    unaffected by the expanded enum, so both are shared with Qwen3-VL.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen3_8DesktopGroundingPointActionSpace
    )
    metadata: LiteCUAMetadata = dataclasses.field(
        default_factory=lambda: LiteCUAMetadata(
            dims=(
                LiteCUAMetadata.Platform.DESKTOP.value,
                LiteCUAMetadata.TaskType.GROUNDING_POINT.value,
            )
        )
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    system_prompt: str | None = GROUNDING_POINT_SYSTEM_PROMPT


@dataclasses.dataclass
class Qwen3_8DesktopUseAdapter(Qwen3_5UseAdapter, key=r"qwen3_8@(desktop|browser)@use"):
    """Desktop+browser ``use`` (multi-step rollout).

    Qwen3.5 rolling-window + image-fold history and XML tool_call wire format,
    with the expanded action enum and the expanded response format. Default
    ``enable_inline_reasoning=False`` → 2-part ``Action:`` + ``<tool_call>``;
    reasoning rides the native ``<think>`` channel gated by
    ``enable_thinking``.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen3_8DesktopActionSpace
    )
    protocol: Qwen3_5HistoryProtocol = dataclasses.field(
        default_factory=Qwen3_5HistoryProtocol
    )
    system_prompt: str | None = QWEN38_USE_SYSTEM_PROMPT
    natively_rendered_extra_tool_names: ClassVar[frozenset[str]] = (
        LiteFinishToolSet.get_tool_names()
    )


# =============================================================================
# Mobile Adapters
# =============================================================================


@dataclasses.dataclass
class Qwen3_8MobileGroundingActionAdapter(
    Qwen3_8BaseAdapter, key="qwen3_8@mobile@grounding.action"
):
    """Mobile grounding/action: full mobile action vocabulary. SFT-replay only."""
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen3_8MobileActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    natively_rendered_extra_tool_names: ClassVar[frozenset[str]] = (
        LiteFinishToolSet.get_tool_names() | LiteAppLaunchToolSet.get_tool_names()
    )


@dataclasses.dataclass
class Qwen3_8MobileGroundingPointAdapter(Qwen3_8BaseAdapter, key="qwen3_8@mobile@grounding.point"):
    """Mobile grounding (single-step click). Same trimmed harness as desktop."""
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen3_8MobileGroundingPointActionSpace
    )
    metadata: LiteCUAMetadata = dataclasses.field(
        default_factory=lambda: LiteCUAMetadata(
            dims=(
                LiteCUAMetadata.Platform.MOBILE.value,
                LiteCUAMetadata.TaskType.GROUNDING_POINT.value,
            )
        )
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    system_prompt: str | None = GROUNDING_POINT_SYSTEM_PROMPT


@dataclasses.dataclass
class Qwen3_8MobileUseAdapter(Qwen3_5UseAdapter, key="qwen3_8@mobile@use"):
    """Mobile ``use`` (multi-step rollout).

    Keeps the Qwen3.5 mobile surface: ``mobile_use`` tool, the shared
    ``USE_SYSTEM_PROMPT`` (inherited — the expanded response format names
    ``call_user``, which the mobile enum does not have), and the desktop
    history protocol. ``smart_resize_enabled=False`` because emulator
    screenshots are already small enough that 32-px rounding would shift
    coordinates.

    Speculative in the same sense as the Qwen3.5 mobile adapter: no upstream
    mobile reference exists for either release.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen3_8MobileActionSpace
    )
    protocol: Qwen3_5HistoryProtocol = dataclasses.field(
        default_factory=Qwen3_5HistoryProtocol
    )
    smart_resize_enabled: bool = False
    natively_rendered_extra_tool_names: ClassVar[frozenset[str]] = (
        LiteFinishToolSet.get_tool_names() | LiteAppLaunchToolSet.get_tool_names()
    )


# =============================================================================
# Pass-through Adapters
# =============================================================================

AgentAdapterRegistry.register(r"qwen3_8@(desktop|browser|mobile)@understanding", AsIsAdapter)
AgentAdapterRegistry.register(r"qwen3_8@(desktop|browser|mobile)@grounding\.bbox", AsIsAdapter)
