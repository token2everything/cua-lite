"""Core agent interfaces for sampling trajectories on CUA-Lite environments.

Public API:
    - ``AgentRegistry``:   registry for agent classes
    - ``BaseAgent``:       abstract base — subclass with ``key=`` to register
    - ``AdapterBasedAgent``: concrete runtime using an adapter + generate_fn
    - ``AutoAdapterAgent``: sugar that auto-resolves the adapter from its key

The adapter-backed runtime follows the same visible loop for every local model
family: observe the env, render a model turn, parse tool calls, step the env,
record feedback, notify hooks, and close the env.

The public agent API spans two data layers: agent-side helper types such as
``PredictResult`` and ``GenerateFn`` live in ``lite.agents.types``; canonical
Lite trajectory records such as ``LiteRLStep``, ``LiteRLSample``, ``LiteSample``,
and ``STATUS_*`` live in ``lite.core``. This module imports both because the
agent loop is the boundary between model code and canonical rollout records.

Prefer importing from the package facade:
    from lite.agents.core.agent import BaseAgent, AgentRegistry
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

import lite.core.messages.image_refs as _image_refs
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.core.adapter import AgentAdapterRegistry, BaseAgentAdapter
from lite.agents.core.agent.hooks import SampleHook, SampleStepData
from lite.agents.core.agent.utils.final import (
    begin_no_tool_call_final,
    mark_no_tool_call_final_result,
)
from lite.agents.core.agent.utils.loop import finalize_lite_sample, mark_steps_truncated
from lite.agents.core.agent.utils.messages import (
    build_initial_user_message,
    build_tool_result_message,
)
from lite.agents.core.agent.utils.tool_results import (
    align_tool_results_to_tool_calls,
    require_model_visible_unpaired_response_feedback,
)
from lite.agents.types import (  # noqa: F401
    AgentMessage,
    AgentStep,
    GenerateFn,
    PredictResult,
)
from lite.core import (
    STATUS_COMPLETED,
    STATUS_TRUNCATED,
    LiteMessage,
    LiteRLSample,
    LiteRLStep,
    LiteSample,
)
from lite.core.errors import ToolCallValidationError
from lite.core.messages import keep_model_visible_content
from lite.core.messages.content import ASSISTANT_ROLE, TEXT_PART
from lite.core.messages.final import (
    pop_model_output_error,
)
from lite.core.messages.turns import count_sample_turns
from lite.core.tools.action_space import LITE_ACTION_BATCH_TOOL_NAMES
from lite.core.tools.calls import stamp_message_tool_call_ids, tool_call_id, tool_call_name
from lite.core.tools.results import LiteToolResult
from lite.gym.base import LiteBaseEnv
from lite.utils.image import decode_image
from lite.utils.registry import BaseRegistry, RegistryKeyed
from lite.utils.timer import Timer

logger = logging.getLogger(__name__)


def _parse_failure_message(response: str) -> dict[str, Any]:
    """Assistant final used when parsing/stamping failed before a Lite turn exists."""
    content = ([{"type": TEXT_PART, "text": response}] if response else [])
    return {"role": ASSISTANT_ROLE, "content": content}


# =============================================================================
# Public API: Registry
# =============================================================================


class AgentRegistry(BaseRegistry["BaseAgent"], cache_by_default=False):
    """Registry for agent classes.

    Terminology:
        agent_id  — model/agent identity, e.g. "qwen3_vl", "ui_tars", "claude"
        agent_key — fully-qualified registry key: "{agent_id}@<metadata dims>",
                    e.g. "qwen3_vl@desktop@use"

    Agents typically require per-call kwargs (processor, generate_fn, etc.),
    so caching is disabled by default.

    Unrecognized kwargs are auto-packed into the agent's ``kwargs`` field.
    Adapter-backed agents treat that field as adapter construction kwargs;
    runtime settings must be consumed before this boundary.

    Example:
        AgentRegistry.get("qwen3_vl@desktop@use",
                          processor=processor,
                          generate_fn=gen_fn,
                          protocol_kwargs={"full_history_size": 4})
    """

    @classmethod
    def get(cls, name: str, *args, cache: bool | None = None, **kwargs) -> BaseAgent:
        return super().get(name, *args, cache=cache, **kwargs)

    @classmethod
    def agent_ids(cls) -> list[str]:
        """All model-ids accepted by :func:`lite.agents.make` (the model catalog).

        The agents analogue of ``gym.registry.env_ids()`` — the ids you pass to
        ``agents.make(model_id, env=...)`` (e.g. ``"Qwen/Qwen3-VL-8B-Instruct"``,
        ``"gpt-5.5"``). Distinct from :meth:`list`, which returns agent registry
        keys (``<agent_id>@<metadata dims...>``).
        """
        from lite.agents.factory import AGENTS

        return sorted(AGENTS)


# =============================================================================
# Base Agent
# =============================================================================


class BaseAgent(RegistryKeyed, ABC):
    """Provider-independent contract implemented by every agent runtime.

    ``predict()`` is the one-turn model boundary: it reads a canonical
    :class:`LiteSample` and returns the next assistant message. ``sample()``
    owns the environment loop: reset, predict, execute tool calls, record
    :class:`LiteRLStep` feedback, notify hooks, and close the environment.
    """

    _registry = AgentRegistry

    @abstractmethod
    async def predict(self, lite_sample: LiteSample) -> LiteMessage:
        """Given the current trajectory state, predict the next assistant message."""
        ...

    @abstractmethod
    async def sample(
        self,
        env: LiteBaseEnv,
        max_steps: int = 10_000,
        hooks: list[SampleHook] | None = None,
    ) -> LiteRLSample:
        """Run a full sampling loop on *env* and return the trajectory.

        *max_steps* is an agent-side safety cap. If it is reached before the env
        terminates or truncates, the returned sample is marked truncated.

        *hooks* receive lifecycle callbacks: ``on_step`` after each env step,
        ``on_complete`` when sampling finishes (or fails).
        """
        ...


# =============================================================================
# Adapter-Based Agent
# =============================================================================


@dataclass
class AdapterBasedAgent(BaseAgent):
    """Agent that uses an adapter + generate function.

    Args:
        generate_fn: ``async (**kwargs) -> {"response": str, ...}``.
            Called with: prompt, images, messages. Must return a dict
            with required "response" key. Optional fields
            (``response_tokens``, ``response_log_probs``,
            ``finish_reason``) are read explicitly into the
            :class:`LiteRLStep`.
        processor: HF processor for building generation prompts via
            ``apply_chat_template``. Required for inference (predict/sample).
        adapter: Converts between CUA-lite and agent-specific formats.
        preserve_raw_response: When True (default), attach the
            ``raw_response`` sidecar (verbatim text + adapter_key fingerprint)
            to valid tool-call assistant messages so next-round replay
            short-circuits to byte-exact. No-tool finals and parse failures stay
            canonical. False → no sidecar (for debug / ablation).
    """

    generate_fn: GenerateFn
    processor: Any = None
    adapter: BaseAgentAdapter = None  # type: ignore[assignment]
    preserve_raw_response: bool = True

    # -- prompt ----------------------------------------------------------------

    def build_generation_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Build a prompt string via ``processor.apply_chat_template``."""
        if self.processor is None:
            raise RuntimeError("agent.processor is not set")
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    # -- predict ---------------------------------------------------------------

    async def predict(self, lite_sample: LiteSample) -> LiteMessage:
        # Standalone predict: process every image fresh. The ``sample()``
        # loop avoids this by maintaining ``processed_images`` across turns.
        processed_images: list[Image.Image | None] = await asyncio.to_thread(
            lambda: [self.adapter.process_image(img) for img in lite_sample.images]
        )
        return (
            await self._predict_with_details(lite_sample, processed_images=processed_images)
        ).lite_message

    def _render_generation_inputs(
        self,
        lite_sample: LiteSample,
        processed_images: list[Image.Image | None],
    ) -> tuple[AgentStep, list[Image.Image], tuple[int, ...], str]:
        """Render, strip model-invisible content, and bind images in prompt order."""
        k = count_sample_turns(lite_sample)
        agent_step = keep_model_visible_content(
            self.adapter.render_step(lite_sample, k, processed_images)
        )
        image_indices = _image_refs.referenced_image_indices_in_message_order(agent_step)
        step_images = _image_refs.referenced_images_in_message_order(
            agent_step,
            processed_images,
        )
        prompt = self.build_generation_prompt(agent_step)
        return agent_step, step_images, image_indices, prompt

    def _parse_generation_response(
        self,
        response: str,
        *,
        call_id_start: int,
    ) -> tuple[AgentMessage, LiteMessage, str | None]:
        """Parse model text, convert to Lite format, and stamp tool-call ids.

        Two model-output-error kinds, one state update each:
        ``parse_error`` means the output cannot become a canonical Lite turn, so
        the single rewrite below replaces both messages with the raw response
        text; ``marker_error`` is the adapter parser's own malformed-output
        signal and keeps the parsed message it was attached to.
        """
        out_agent_message = self.adapter.parse_raw_assistant_response(response)
        parse_error: str | None = None
        marker_error: str | None = None
        try:
            out_lite_message = self.adapter.convert_message_from_agent(out_agent_message)
        except ModelToolCallParseError as e:
            # Action-space parsers use this named error for malformed model
            # actions. Incidental TypeError/ValueError bugs still propagate.
            parse_error = str(e)
        else:
            # Adapter-owned parse markers are transient side channels; pop both
            # carriers so neither message keeps one. The no-tool final
            # classifier consumes the string separately.
            lite_error = pop_model_output_error(out_lite_message)
            agent_error = pop_model_output_error(out_agent_message)
            marker_error = lite_error or agent_error
            if (
                not marker_error
                and out_agent_message.get("tool_calls")
                and not out_lite_message.get("tool_calls")
            ):
                parse_error = (
                    "model emitted tool_call(s), but none converted to canonical Lite tool_calls"
                )
            else:
                try:
                    # Lite owns call ids: whatever id the model emitted is
                    # discarded and replaced by ``call_{call_id_start + i:04d}``,
                    # which the sample() loop advances per turn, so ids stay
                    # unique and result-pairable across the trajectory. Stamping
                    # is also the canonical Lite call-shape check.
                    stamp_message_tool_call_ids(
                        out_lite_message,
                        start=call_id_start,
                        preserve=False,
                    )
                except ToolCallValidationError as e:
                    # Stamping is the canonical Lite message boundary. Shape
                    # errors here are model-output errors; container/type
                    # mistakes stay loud.
                    parse_error = str(e)

        if parse_error is not None:
            # Sole owner of the parse-failure turn shape: the model's raw text
            # becomes a plain assistant final that the loop terminates on.
            out_agent_message = _parse_failure_message(response)
            out_lite_message = dict(out_agent_message)
        model_output_error = parse_error or marker_error

        # Only valid tool-call turns can replay byte-exact raw provider output.
        if (
            self.preserve_raw_response
            and out_lite_message.get("tool_calls")
            and not model_output_error
        ):
            out_lite_message["raw_response"] = {
                "text": response,
                "adapter_key": self.adapter.get_registry_key(),
            }

        return out_agent_message, out_lite_message, model_output_error

    async def _predict_with_details(
        self,
        lite_sample: LiteSample,
        *,
        processed_images: list[Image.Image | None],
        call_id_start: int = 0,
    ) -> PredictResult:
        """Render the current turn via :meth:`render_step`, generate, parse.

        ``processed_images`` is the trajectory's processed-image list —
        one entry per image in ``lite_sample.images``. The agent's
        :meth:`sample` loop maintains it across turns (one
        ``adapter.process_image`` call per new image); standalone
        ``predict()`` callers compute it from scratch upstream.
        """
        t = Timer()

        # Render, strip model-invisible side channels, order images, and build
        # the prompt off the event loop. Image order must match prompt
        # placeholders so multimodal processors bind pixels to the right tokens.
        agent_step, step_images, image_indices, prompt = await asyncio.to_thread(
            self._render_generation_inputs,
            lite_sample,
            processed_images,
        )
        t.mark("render")

        gen_response: dict[str, Any] = await self.generate_fn(
            prompt=prompt,
            images=step_images,
            messages=agent_step,
        )
        t.mark("generate")

        response: str = gen_response["response"]

        out_agent_message, out_lite_message, model_output_error = await asyncio.to_thread(
            self._parse_generation_response,
            response,
            call_id_start=call_id_start,
        )
        t.mark("parse")

        # Map finish_reason → status. Anything truncation-shaped (length /
        # context-length-exceeded) maps to TRUNCATED; everything else
        # (stop / tool_calls / completed) maps to COMPLETED. ABORTED is
        # set by the rollout loop on AbortError, not here.
        finish_reason = (gen_response.get("finish_reason") or "stop").lower()
        if finish_reason in ("length", "max_tokens", "context_length_exceeded"):
            status = STATUS_TRUNCATED
        else:
            status = STATUS_COMPLETED

        rl_step = LiteRLStep(
            prompt=prompt,
            image_indices=image_indices,
            response=response,
            response_tokens=list(gen_response.get("response_tokens") or []),
            response_log_probs=list(gen_response.get("response_log_probs") or []),
            reward=0.0,  # filled in by sample() loop after env.step
            status=status,
        )

        logger.info("⏱ predict: %s n_images=%d", t, len(step_images))
        return PredictResult(
            lite_message=out_lite_message,
            agent_message=out_agent_message,
            step=rl_step,
            model_output_error=model_output_error,
        )

    # -- sample ----------------------------------------------------------------

    async def _append_initial_observation(
        self,
        lite_sample: LiteSample,
        lite_rl_sample: LiteRLSample,
        *,
        image_bytes: bytes | None,
        instruction: str,
        metadata: dict[str, Any] | None,
    ) -> tuple[Image.Image | None, int | None]:
        """Append the reset observation and process its image, if any."""
        image = None
        image_index: int | None = None
        if image_bytes:
            image = await asyncio.to_thread(decode_image, image_bytes)
            lite_sample.images.append(image)
            lite_rl_sample.processed_images.append(
                await asyncio.to_thread(self.adapter.process_image, image)
            )
            image_index = len(lite_sample.images) - 1
        lite_sample.messages.append(
            build_initial_user_message(
                instruction,
                has_image=(image_index is not None),
                metadata=metadata,
            )
        )
        return image, image_index

    async def _append_tool_result_messages(
        self,
        lite_sample: LiteSample,
        lite_rl_sample: LiteRLSample,
        tool_results: list[LiteToolResult],
        *,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> tuple[Image.Image | None, int | None]:
        """Append env tool feedback; return the first model-visible result image."""
        hook_image: Image.Image | None = None
        hook_image_index: int | None = None
        # Keyed by call id, not a bare id set: the adapter's visibility hook is
        # handed the CALL so an override can decide from the batch's own child
        # actions, not just from how many frames came back.
        action_batch_calls_by_id = {
            call_id: call
            for call in tool_calls or []
            if (call_id := tool_call_id(call))
            and tool_call_name(call) in LITE_ACTION_BATCH_TOOL_NAMES
        }
        for tool_result in tool_results:
            stored_images: list[tuple[int, Image.Image]] = []
            for image_bytes in tool_result.images:
                result_image = await asyncio.to_thread(decode_image, image_bytes)
                image_index = len(lite_sample.images)
                lite_sample.images.append(result_image)
                lite_rl_sample.processed_images.append(None)
                stored_images.append((image_index, result_image))

            # Store every env image durably. For action-batch calls the ADAPTER
            # owns which stored frames are model-visible next turn; the base
            # default is the final one. Dropped frames keep their indices and
            # their ``processed_images`` slots stay None, so the visible
            # indices jump over them rather than renumbering.
            action_batch_call = action_batch_calls_by_id.get(tool_result.tool_call_id)
            if action_batch_call is not None:
                selected = self.adapter.select_action_batch_image_indices(
                    tool_call=action_batch_call,
                    tool_result=tool_result,
                    result_image_indices=tuple(index for index, _ in stored_images),
                )
                selected_set = set(selected)
                model_visible_images = [
                    pair for pair in stored_images if pair[0] in selected_set
                ]
            elif tool_result.tool_call_id is None and stored_images:
                model_visible_images = stored_images[-1:]
            else:
                model_visible_images = stored_images
            model_visible_image_indices = tuple(
                image_index
                for image_index, _result_image in model_visible_images
            )
            for image_index, result_image in model_visible_images:
                lite_rl_sample.processed_images[image_index] = await asyncio.to_thread(
                    self.adapter.process_image,
                    result_image,
                )
            if hook_image is None and model_visible_images:
                hook_image_index, hook_image = model_visible_images[0]

            message = build_tool_result_message(
                tool_result.tool_call_id,
                model_visible_image_indices,
                tool_result.text,
                tool_result.metadata,
                error=tool_result.error,
            )
            if message is not None:
                lite_sample.messages.append(message)
        return hook_image, hook_image_index

    async def sample(
        self,
        env: LiteBaseEnv,
        max_steps: int = 10_000,
        hooks: list[SampleHook] | None = None,
    ) -> LiteRLSample:
        hooks = hooks or []
        lite_rl_sample: LiteRLSample | None = None
        completed = False

        try:
            t = Timer()
            reset_observation = await env.reset()
            t.mark("reset")

            image_bytes = reset_observation.image
            instruction = reset_observation.text
            metadata = reset_observation.metadata  # reset-obs metadata -> turn-0 msg
            # The task instruction is mandatory; the image is optional for
            # text-only environments such as AXTree-style browser observations.
            if not image_bytes and not instruction:
                raise RuntimeError("Env returned neither image nor text on env.reset()")
            if not instruction:
                raise RuntimeError(
                    "No instruction returned on env.reset() (observation.text is None/empty)"
                )

            # ``LiteRLSample`` wraps the mutable ``LiteSample`` used by
            # render_step; processed images live beside it for tokenization.
            lite_sample: LiteSample = LiteSample(metadata=env.metadata)
            lite_rl_sample = LiteRLSample(lite_sample=lite_sample)
            # State carried across turns: the previous turn's env feedback plus
            # the calls it answers (both None on a terminal turn), and the
            # trajectory-wide tool-call id counter that keeps every persisted
            # call id unique so results can pair against it. A content-only
            # answer-feedback turn has feedback but no persisted calls; its
            # results append as role:user observations on the next iteration.
            pending_tool_results: list[LiteToolResult] | None = None
            pending_tool_calls: list[dict[str, Any]] | None = None
            next_call_id = 0

            for step_idx in range(max_steps):
                t = Timer()

                # 1. Prepare input: append the reset obs or the prior turn's env
                # feedback. ``image``/``image_index`` are the observation this
                # turn's model call sees, and are what hooks report for the step.
                if step_idx == 0:
                    image, image_index = await self._append_initial_observation(
                        lite_sample,
                        lite_rl_sample,
                        image_bytes=image_bytes,
                        instruction=instruction,
                        metadata=metadata,
                    )
                elif pending_tool_results is not None:
                    image, image_index = await self._append_tool_result_messages(
                        lite_sample,
                        lite_rl_sample,
                        pending_tool_results,
                        tool_calls=pending_tool_calls,
                    )
                else:
                    raise RuntimeError("Internal error: no pending tool results for continued turn")
                observe_s = t.mark("observe")
                for hook in hooks:
                    if publish := getattr(hook, "on_trajectory", None):
                        await asyncio.to_thread(publish, lite_sample)

                # 2. Call provider and parse output: render the turn, generate,
                # and convert the model text into a canonical Lite message whose
                # call ids continue this trajectory's sequence.
                predict_result = await self._predict_with_details(
                    lite_sample,
                    processed_images=lite_rl_sample.processed_images,
                    call_id_start=next_call_id,
                )
                next_call_id += len(predict_result.lite_message.get("tool_calls") or [])
                predict_s = t.mark("predict")

                # Record trajectory: the assistant turn and its RL step are
                # persisted before env.step, so a failing step still leaves the
                # model output on the sample.
                lite_sample.messages.append(predict_result.lite_message)
                lite_rl_sample.steps.append(predict_result.step)

                # 3. Step env: execute the model's tool calls, or the runtime
                # response/final action of a no-tool-call turn.
                assistant_tool_calls = predict_result.lite_message.get("tool_calls") or []
                if assistant_tool_calls:
                    executed_actions = assistant_tool_calls
                    step_result = await env.step(executed_actions)
                else:
                    # Zero parsed tool calls go through the shared final
                    # classifier. Deliberate content-only answers follow the
                    # env's returned terminal flags; parse failures still force
                    # terminal because they are malformed model output.
                    final = begin_no_tool_call_final(
                        predict_result.lite_message,
                        model_output_error=predict_result.model_output_error,
                        step=predict_result.step,
                    )
                    lite_sample.messages[-1] = final.message
                    predict_result.lite_message = final.message
                    executed_actions = final.actions
                    step_result = await env.step(executed_actions)
                    step_result = mark_no_tool_call_final_result(
                        step_result,
                        final,
                    )
                act_s = t.mark("act")

                logger.info("⏱ sample step %d: %s", step_idx, t)

                # 4. Prepare next turn: decide whether this turn's tool feedback
                # is persisted now (terminal) or carried into the next prompt.
                # This branch and the accounting below stay inline rather than
                # calling ``record_lite_env_result``: only this loop keeps the
                # aligned results as the NEXT turn's observation and maintains
                # ``processed_images`` while appending them.
                env_terminal_step = step_result.terminated or step_result.truncated
                agent_truncated = step_idx + 1 >= max_steps and not env_terminal_step
                is_terminal_step = (
                    (not assistant_tool_calls and step_result.terminated)
                    or env_terminal_step
                    or agent_truncated
                )
                aligned_results: list[LiteToolResult] = []
                if assistant_tool_calls:
                    aligned_results = align_tool_results_to_tool_calls(
                        assistant_tool_calls,
                        step_result.results,
                        require_all=not env_terminal_step,
                    )
                    step_result.results = aligned_results
                else:
                    aligned_results = list(step_result.results)
                if is_terminal_step:
                    if assistant_tool_calls and aligned_results:
                        # Persist final call feedback; hooks still describe the
                        # image observed at the start of this step.
                        await self._append_tool_result_messages(
                            lite_sample,
                            lite_rl_sample,
                            aligned_results,
                            tool_calls=assistant_tool_calls,
                        )
                    pending_tool_results = None
                    pending_tool_calls = None
                else:
                    if not aligned_results:
                        raise RuntimeError(
                            "Env returned no tool results after non-terminal env.step()"
                        )
                    if not assistant_tool_calls:
                        require_model_visible_unpaired_response_feedback(aligned_results)
                    # These become the next turn's model-visible observation;
                    # the calls travel with them so action-batch feedback can be
                    # narrowed to its final screenshot when it is appended. For
                    # content-only continuations there are no persisted calls,
                    # so feedback appends as unpaired role:user messages.
                    pending_tool_results = aligned_results
                    pending_tool_calls = assistant_tool_calls

                # 5. Record trajectory accounting: reward, terminal flags, and
                # the RL step status for this turn.
                if step_result.reward is not None:
                    lite_rl_sample.episode_return += step_result.reward
                    predict_result.step.reward = float(step_result.reward)
                lite_rl_sample.terminated = step_result.terminated
                lite_rl_sample.truncated = step_result.truncated or agent_truncated
                if agent_truncated:
                    step_result.truncated = True
                if lite_rl_sample.truncated and predict_result.step.status == STATUS_COMPLETED:
                    # Env truncation overrides COMPLETED; preserve TRUNCATED
                    # / ABORTED if predict already set them.
                    predict_result.step.status = STATUS_TRUNCATED

                # 6. Notify hooks off-loop; loggers may encode images and write
                # files. ``image`` is the observation this turn's model call
                # saw, and ``executed_actions`` is what env.step actually ran.
                step_data = SampleStepData(
                    step_idx=step_idx,
                    image=image,
                    predict_result=predict_result,
                    step_result=step_result,
                    actions=executed_actions,
                    current_image_index=image_index,
                    timings={"observe": observe_s, "predict": predict_s, "act": act_s},
                )
                for hook in hooks:
                    await asyncio.to_thread(hook.on_step, step_data)
                if is_terminal_step:
                    for hook in hooks:
                        if publish := getattr(hook, "on_trajectory", None):
                            await asyncio.to_thread(publish, lite_sample)

                # 7. Stop once the rollout is terminal or truncated.
                if lite_rl_sample.terminated or lite_rl_sample.truncated:
                    break
            else:
                # max_steps exhausted without a terminal turn (reachable only
                # when max_steps <= 0, since step_idx + 1 >= max_steps already
                # sets agent_truncated on the last iteration).
                lite_rl_sample.truncated = True
                mark_steps_truncated(lite_rl_sample.steps)

            completed = True
            return lite_rl_sample
        finally:
            if not completed and lite_rl_sample is not None:
                for hook in hooks:
                    if publish := getattr(hook, "on_trajectory", None):
                        try:
                            await asyncio.to_thread(publish, lite_rl_sample.lite_sample)
                        except Exception:
                            logger.warning("Failed to publish interrupted trajectory", exc_info=True)
            await finalize_lite_sample(
                hooks=hooks,
                result=lite_rl_sample,
                completed=completed,
                env=env,
                logger=logger,
            )


# =============================================================================
# Auto-Adapter Agent (common base for registry-backed agents)
# =============================================================================


@dataclass
class AutoAdapterAgent(AdapterBasedAgent):
    """Agent that auto-resolves its adapter from AgentAdapterRegistry.

    Subclass with ``key=`` to register a new agent. All common fields
    (generate_fn, processor, overrides) are inherited — subclasses only
    need to add model-specific behavior if any.

    Example:
        @dataclass
        class MyAgent(AutoAdapterAgent, key="mymodel@desktop@use"):
            pass  # inherits everything

        agent = AgentRegistry.get("mymodel@desktop@use",
                                  processor=proc, generate_fn=fn,
                                  protocol_kwargs={"full_history_size": 2})
    """

    # Catch-all for kwargs not recognized as agent fields.
    # Entries are splatted to AgentAdapterRegistry.get() in __post_init__, which
    # auto-separates into adapter fields vs adapter.kwargs and fails on unknowns.
    kwargs: dict[str, Any] = field(default_factory=dict)

    def _adapter_kwargs(self) -> dict[str, Any]:
        """Return a copy of the packed adapter construction kwargs."""
        return dict(self.kwargs)

    def __post_init__(self):
        self.adapter = AgentAdapterRegistry.get(
            self.get_registry_key(),
            **self._adapter_kwargs(),
        )
