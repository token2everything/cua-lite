"""GPT teacher agent for distillation data collection (model-side).

A thin :class:`GPTDesktopUseAgent` subclass for collecting *better* teacher
trajectories to distill into small students. It differs from the base GPT agent
in exactly two, fully **YAML-driven** ways — the base agent stays untouched:

  1. **Structured response contract.** It appends a two-line contract built from
     the config strings ``inline_reasoning_instruction`` /
     ``action_description_instruction`` (so the *requirement* lives in YAML, not code):

         Thought: {inline_reasoning_instruction}
         Action:  {action_description_instruction}
         <tool_call>

  2. **Controlled inline_reasoning.** It post-processes the model's reply,
     splitting the ``Thought:`` / ``Action:`` labels into an ``inline_reasoning``
     part (the Thought) and an ``action_description`` part (the imperative). The
     student therefore distills a *prompt-controlled* reasoning channel instead of
     the OpenAI ``reasoning.summary`` (a semi-stochastic post-hoc summary that
     ignores formatting instructions — see the base agent's ``reasoning_summary``
     path, kept as the default when this subclass is NOT used).

WHY a subclass: the API ``reasoning.summary`` cannot be steered by prompt — pushing
structure onto it makes it vanish — so the controlled reasoning must be produced as
ordinary instruction-following output and parsed back out. That distillation-only
behavior does not belong in the production GPT agent, so it lives here and is opted
into per-run via ``agent_id: gpt.teacher`` (the factory's yaml-driven ``agent_id``
override, same mechanism BrowserGym uses for ``qwen3_vl.base``).

Run (collection):
    # in a gpt collect yaml:
    #   agent_id: gpt.teacher
    #   agent_kwargs:
    #     inline_reasoning_instruction: "...observe-first reasoning..."
    #     action_description_instruction: "one short imperative naming the next UI action."
    #     api_kwargs: { reasoning_summary: null }
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from lite.agents.models.gpt.agent import GPTDesktopUseAgent
from lite.agents.models.gpt.utils.parse import GPTParsedOutput
from lite.core import (
    LiteMessage,
)

# Sensible defaults for the three prose pieces of the response contract — each
# overridable verbatim from yaml ``agent_kwargs``. Two describe the *content* of a line
# (``inline_reasoning`` → the ``Thought:`` line; ``action_description`` → the ``Action:``
# line); ``reasoning_cadence`` is the adaptive policy for WHEN to emit the Thought.
# Positive framing (state what to DO).
DEFAULT_INLINE_REASONING_INSTRUCTION = (
    "one sentence that reasons from what is visible on the screen right now — name the "
    "relevant on-screen elements you can see, note whether your previous action produced "
    "its intended visible effect, and reason toward the next step (brief forward planning "
    "is fine when it stays grounded in what you actually observe)"
)
DEFAULT_ACTION_DESCRIPTION_INSTRUCTION = (
    'one short imperative naming the single UI action you will take (e.g. "Right-click the Sheet1 tab.")'
)
# Adaptive cadence — WHEN to emit the Thought (matches the student's "(Optional) Thought"
# recipe: reason on decision/verification steps, skip routine mechanical ones). The Action
# line is always required regardless of cadence.
DEFAULT_REASONING_CADENCE = (
    "Add a 'Thought:' line above the Action whenever the step warrants reasoning — "
    "deciding what to do next from the current screen, or verifying that a result or "
    "the overall goal is now correct (always include the Thought when you are finishing, "
    "to confirm the goal is met). Skip the Thought on routine mechanical follow-through steps."
)


def _split_thought_action(text: str) -> tuple[str, str]:
    """Split a ``Thought: ...\\nAction: ...`` teacher reply into ``(thought, action)``.

    Label-driven and multi-line tolerant: lines are routed to whichever of the two
    labels is currently open. Graceful fallbacks: a reply with no labels is treated
    wholly as the action; ``Action:`` missing leaves an empty action (the tool_call
    still carries the move)."""
    thought_parts: list[str] = []
    action_parts: list[str] = []
    leftover: list[str] = []
    mode: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("thought:"):
            mode = "t"
            body = stripped[len("thought:"):].strip()
            if body:
                thought_parts.append(body)
        elif low.startswith("action:"):
            mode = "a"
            body = stripped[len("action:"):].strip()
            if body:
                action_parts.append(body)
        elif mode == "t":
            thought_parts.append(stripped)
        elif mode == "a":
            action_parts.append(stripped)
        else:
            leftover.append(stripped)
    thought = " ".join(p for p in thought_parts if p).strip()
    action = " ".join(p for p in action_parts if p).strip()
    if not action:
        action = " ".join(p for p in leftover if p).strip()
    if not thought and not action:
        action = text.strip()
    return thought, action


@dataclass
class GPTTeacherAgent(GPTDesktopUseAgent, key=r"gpt\.teacher(@(desktop|browser)@use)?"):
    """GPT teacher that emits a YAML-defined ``Thought:`` + ``Action:`` per step,
    distilled as ``inline_reasoning`` + ``action_description`` (not the API summary)."""

    # The three prose pieces of the response contract — all defined in yaml ``agent_kwargs``.
    # ``inline_reasoning_instruction`` / ``action_description_instruction`` are the *content*
    # of the ``Thought:`` / ``Action:`` lines; ``reasoning_cadence`` is the adaptive policy
    # for WHEN to emit the Thought (the Action line is always required).
    inline_reasoning_instruction: str = DEFAULT_INLINE_REASONING_INSTRUCTION
    action_description_instruction: str = DEFAULT_ACTION_DESCRIPTION_INSTRUCTION
    reasoning_cadence: str = DEFAULT_REASONING_CADENCE

    # NOTE on the API reasoning summary: ``_relabel_thought_action`` always discards any
    # summary-derived ``inline_reasoning`` (the controlled ``Thought:`` line replaces it),
    # so correctness never depends on the summary. To also avoid generating one that gets
    # thrown away, set ``api_kwargs: { reasoning_summary: null }`` in the yaml (the base
    # leaves an explicit ``None`` untouched, and ``_build_reasoning`` then omits it).

    def _structured_output_instruction(self) -> str:
        """Assemble the per-step response contract from its three yaml-overridable parts:
        ``reasoning_cadence`` (when to emit the Thought), ``inline_reasoning_instruction``
        (what the Thought says) and ``action_description_instruction`` (what the Action
        says). The Action line + tool call are always required; ``_split_thought_action``
        handles a skipped Thought gracefully (action_description only)."""
        return (
            "For every tool call, output an 'Action:' line and then make the tool call. "
            f"{self.reasoning_cadence}\n"
            f"Thought (when used): {self.inline_reasoning_instruction}\n"
            f"Action: {self.action_description_instruction}"
        )

    def _effective_system_prompt(self) -> str | None:
        # Base handles system_prompt (+ suffix). This subclass produces (and parses) its
        # own Action line via the Thought:/Action: contract — the base agent has no
        # action-description mechanism of its own.
        base = super()._effective_system_prompt() or ""
        suffix = self._structured_output_instruction()
        combined = (base + ("\n" if base and suffix else "") + suffix).strip()
        return combined or None

    def _parse_output_items(
        self,
        output_items: list[dict[str, Any]],
        *,
        resolution: tuple[int, int],
        active_provider_tool_names: frozenset[str] | None = None,
        call_id_start: int = 0,
    ) -> GPTParsedOutput:
        # Override the base parse seam so the ``Thought:`` / ``Action:`` reply is
        # relabelled wherever the message is produced. Only the Lite message changes;
        # the provider->canonical provenance the loop feeds back is the base's.
        # The signature must track the base seam EXACTLY — ``sample()`` always calls it
        # with ``call_id_start=next_call_id`` (lite/agents/models/gpt/agent.py), so a
        # narrower override raises TypeError on the first model reply of every run.
        parsed_output = super()._parse_output_items(
            output_items,
            resolution=resolution,
            active_provider_tool_names=active_provider_tool_names,
            call_id_start=call_id_start,
        )
        return replace(
            parsed_output,
            message=self._relabel_thought_action(parsed_output.message),
        )

    def _relabel_thought_action(self, msg: LiteMessage) -> LiteMessage:
        """Rewrite the reply's text/action_description content into a grounded
        ``inline_reasoning`` (Thought) + ``action_description`` (Action) pair. Any
        summary-sourced ``inline_reasoning`` is dropped — the controlled Thought wins.

        Only for a turn that actually carries ``tool_calls``: the Thought/Action
        contract narrates an action. A no-tool-call turn is the model's
        text-oriented termination (or a parse failure), and its prose must stay
        plain ``text`` so the loop's shared classifier
        (:func:`lite.core.messages.final.no_tool_call_final_text`) sees it."""
        if not msg.get("tool_calls"):
            return msg
        content = msg.get("content") or []
        carried: list[dict] = []
        raw = ""
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in ("text", "action_description"):
                raw = (raw + "\n" + part.get("text", "")).strip() if raw else part.get("text", "")
            elif ptype == "inline_reasoning":
                continue  # drop the summary-derived reasoning; replaced below
            else:
                carried.append(part)
        thought, action = _split_thought_action(raw) if raw else ("", "")
        new_content: list[dict] = []
        if thought:
            new_content.append({"type": "inline_reasoning", "text": thought})
        if action:
            new_content.append({"type": "action_description", "text": action})
        new_content.extend(carried)
        return {**msg, "content": new_content}
