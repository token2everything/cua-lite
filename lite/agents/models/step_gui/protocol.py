"""StepGUI history protocol (``step_gui.history``).

StepGUI's rolling-summary style: each assistant response ends with a
``summary:`` KV that cumulatively describes all prior actions; only the
current screenshot + the latest rolling summary are sent on each step.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from lite.agents.core.protocol import TurnWindowProtocol
from lite.agents.core.protocol.window import join_role_tool_text
from lite.core import (
    LiteMessage,
)
from lite.core.messages import (
    HISTORY_SUMMARY_PART,
    TEXT_PART,
    USER_ROLE,
)
from lite.core.messages.content import first_image_content_part


def _extract_summary_field(message: LiteMessage | None) -> str:
    """Extract StepGUI's rolling ``summary`` from an assistant message.

    StepGUI's raw output puts ``summary:累计历史总结`` in the tab-separated
    KV line; the adapter parses it into a ``HistorySummaryContent`` part
    inside ``content``. This helper picks it up from there.
    """
    if not message:
        return ""
    for part in message.get("content") or []:
        if part["type"] == HISTORY_SUMMARY_PART and part["text"]:
            return part["text"].strip()
    return ""


@dataclasses.dataclass
class StepGUIHistoryProtocol(TurnWindowProtocol, key="step_gui.history"):
    """
    History protocol matching the vanilla StepGUI (parser_0920_summary) logic.

    StepGUI uses a *rolling summary* approach:

    * Each assistant response ends with a ``summary:`` field that contains a
      **cumulative** natural-language description of all actions taken so far.
    * At each new turn the ``summary`` from the **last** assistant response is
      extracted and injected into the user prompt as ``已知已经执行过的历史动作如下:``.
    * Only the current screenshot is sent (no image history).

    This is a Markov-style protocol: the model is trained to produce a
    self-contained summary each step, so we only need the latest one — no
    concatenation of per-step descriptions.

    Reference:
        gelab-zero/copilot_tools/parser_0920_summary.py
        (``env2messages4ask`` lines 320-324 + ``make_status_prompt``)
    """

    def _select_messages(
        self,
        content: list[LiteMessage],
        turns: list[dict[str, Any]],
    ) -> list[LiteMessage]:
        if not turns:
            return list(content)
        else:
            return self._build_current_turn(turns)

    def _build_current_turn(
        self,
        turns: list[dict[str, LiteMessage | None]],
    ) -> list[LiteMessage]:
        """Collapse all history into the rolling summary and keep only the
        current screenshot, matching ``make_status_prompt``.

        Handles two call contexts:

        * **Fresh rollout** — the last turn has a user message but no
          assistant. The rolling summary comes from the previous assistant
          (``turns[-2]`` or earlier).
        * **SFT unroll** — the ``adapter.unroll`` skeleton (called from
          ``export_sft``) feeds in a trajectory whose last turn includes
          the prediction target assistant. The
          target is appended to the protocol output as the SFT label, and
          the rolling summary is pulled from turns **before** the target so
          that the target's own ``summary`` does not leak back into its
          prompt.
        """

        # Task instruction comes from the FIRST user message (the goal given
        # at the start of the trajectory).
        task = self._extract_instruction(turns[0])

        # If the last turn has an assistant, treat it as the SFT target.
        last_turn = turns[-1]
        target_assistant = last_turn.get("assistant")

        # Extract the cumulative summary from the most recent assistant
        # BEFORE the target (or the most recent overall for fresh rollout).
        search_turns = turns[:-1] if target_assistant else turns
        summary_history = ""
        for turn in reversed(search_turns):
            if turn.get("assistant"):
                summary_history = _extract_summary_field(turn["assistant"])
                break

        history_display = summary_history if summary_history.strip() else "暂无历史操作"

        # Current screenshot comes from the latest observation in the last turn.
        current_image = None
        current_observations = last_turn["observations"]
        current_tool_text = join_role_tool_text(current_observations)
        for user_msg in reversed(current_observations):
            current_image = first_image_content_part(user_msg)
            if current_image is not None:
                break

        # Assemble a single user message mirroring make_status_prompt
        # (parser_0920_summary.py:71-97).
        # Newline count here matches reference `make_status_prompt` exactly:
        # reference concatenates `task + "指令结束\\n\\n"` inside an f-string
        # that already contains a trailing `\\n` before the next line, so there
        # are 3 newlines between `指令结束` and `已知已经执行过的历史动作`.
        user_content: list[dict] = [
            {
                "type": TEXT_PART,
                "text": (
                    f"\n已知用户指令为：{task}指令结束\n\n\n"
                    f"已知已经执行过的历史动作如下：{history_display}\n"
                    f"当前手机屏幕截图如下：\n"
                ),
            },
        ]
        if current_image is not None:
            user_content.append(current_image)
        if current_tool_text is not None:
            user_content.append({"type": TEXT_PART, "text": f"\n\n{current_tool_text}"})
        user_content.append({
            "type": TEXT_PART,
            "text": (
                "\n\n在执行操作之前，请务必回顾你的历史操作记录和限定的动作空间，"
                "先进行思考和解释然后输出动作空间和对应的参数：\n"
                "1. 思考（THINK）：在 <THINK> 和 </THINK> 标签之间。\n"
                "2. 解释（explain）：在动作格式中，使用 explain: 开头，简要说明当前动作的目的和执行方式。\n"
                "在执行完操作后，请输出执行完当前步骤后的新历史总结。\n"
                "输出格式示例：\n"
                "<THINK> 思考的内容 </THINK>\n"
                "explain:解释的内容\taction:动作空间和对应的参数\tsummary:执行完当前步骤后的新历史总结\n"
            ),
        })

        result: list[LiteMessage] = [{"role": USER_ROLE, "content": user_content}]
        if target_assistant:
            result.append(target_assistant)
        return result
