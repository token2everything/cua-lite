"""Gemini computer-use agents (desktop / browser).

Talks to Gemini's **native** ``generateContent`` REST surface, not the
OpenAI-compatible chat-completions path: litellm has no representation for the
``computerUse`` tool, so that translation drops it entirely -- and with it
``environment``, which is what selects the desktop / browser / mobile verb set.

Coordinates are already on the canonical 0-1000 grid, so unlike the Claude
desktop agent there is no ``scale_x``/``scale_y`` stage and the tool
declaration carries no display dimensions.

Usage::

    agent = GeminiDesktopUseAgent(
        model_id="gemini-3.6-flash",
        api_base="https://<proxy>",
        api_key="...",
        metadata=env.metadata,
    )
    sample = await agent.sample(env, max_steps=30)
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

from lite.agents.core.agent import BaseAgent
from lite.agents.core.agent.hooks import SampleHook
from lite.agents.core.agent.utils.image_markers import (
    PROVIDER_VISIBLE_IMAGE_INDEX_MARKER,
    mark_provider_visible_image_index,
)
from lite.agents.core.agent.utils.loop import (
    append_lite_rl_step,
    build_lite_rl_sample,
    dispatch_lite_step_hooks,
    execute_lite_turn,
    finalize_lite_sample,
    mark_steps_truncated,
    record_lite_env_result,
)
from lite.agents.core.agent.utils.retry import acompletion_with_retry
from lite.agents.models.gemini.action_space import (
    GeminiDesktopActionSpace,
    GeminiMobileActionSpace,
)
from lite.agents.models.gemini.utils.history import (
    append_gemini_provider_feedback,
    append_provider_assistant_message,
    filter_to_n_most_recent_images,
    image_b64,
    strip_images,
)
from lite.agents.models.gemini.utils.loop import _append_gemini_terminal_tool_feedback
from lite.agents.models.gemini.utils.parse import parse_response_with_provenance
from lite.agents.models.gemini.utils.transport import agenerate_content
from lite.core import LiteMessage, LiteRLSample, LiteRLStep, LiteSample
from lite.core.messages.final import pop_model_output_error
from lite.core.metadata import LiteBaseMetadata, LiteCUAMetadata
from lite.core.tools.schemas import tool_schema_name
from lite.gym.base import LiteBaseEnv
from lite.utils.image import decode_image, normalize_resolution
from lite.utils.timer import timed

logger = logging.getLogger(__name__)

#: ``append_lite_rl_step`` lowercases ``finish_reason`` before the membership
#: test, so this set MUST be lowercase. Gemini's native values are UPPERCASE
#: ("STOP", "MAX_TOKENS"); listing them verbatim would never match and every
#: step would silently record STATUS_COMPLETED.
_TRUNCATION_FINISH_REASONS = ("max_tokens",)

GEMINI_API_KWARGS_DEFAULTS: dict[str, Any] = {
    "thinking_level": "medium",
}


@dataclass
class _GeminiBaseAgent(BaseAgent):
    """Shared request plumbing for the Gemini computer-use agents."""

    _API_KWARGS_DEFAULTS: ClassVar[dict[str, Any]] = GEMINI_API_KWARGS_DEFAULTS
    _RETRY_LOG_NAME: ClassVar[str] = "gemini.generateContent"
    #: Mobile passes named non-native calls through; desktop rejects them.
    _PASS_THROUGH_UNKNOWN: ClassVar[bool] = False

    model_id: str = "gemini-3.6-flash"
    api_kwargs: dict[str, Any] = field(default_factory=dict)
    api_key: str | None = None
    #: Proxy root. No usable default exists, but it is NOT validated here:
    #: ``make(model_id, env=env)`` constructs agents without it in tests, so the
    #: failure belongs at the first request instead (transport raises).
    api_base: str | None = None
    api_timeout_s: float = 120.0
    api_retry_max: int = 3
    api_retry_base_delay: float = 0.5
    api_retry_max_delay: float = 60.0

    system_prompt: str | None = None
    system_prompt_suffix: str | None = None
    #: Pre-send resize target. Byte budget only -- it does NOT participate in
    #: coordinate conversion, which is identity on this provider.
    resolution: tuple[int, int] | None = None
    only_n_most_recent_images: int | None = None
    image_truncation_threshold: int = 1

    tool_execution_timeout_s: float | None = None
    metadata: LiteBaseMetadata | None = None

    action_space: Any = field(default_factory=GeminiDesktopActionSpace)

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = LiteCUAMetadata()
        if self.api_kwargs is None:
            self.api_kwargs = {}
        for key, value in self._API_KWARGS_DEFAULTS.items():
            self.api_kwargs.setdefault(key, value)
        # A bespoke API loop constructs no adapter, so it does not inherit
        # ``BaseAgentAdapter``'s coercion -- same rule, one owner.
        self.resolution = normalize_resolution(self.resolution)
        # NOTE: platform is passed to the action space per call, never written
        # onto it. ``AgentRegistry.get`` does not back-fill ``.platform`` the
        # way ``ActionSpaceRegistry.get`` does, so relying on it would read
        # 'desktop' even for ``gemini@browser@use``.

    def _cua_metadata(self) -> LiteCUAMetadata:
        if not isinstance(self.metadata, LiteCUAMetadata):
            raise TypeError(
                f"{type(self).__name__} requires LiteCUAMetadata for CUA tool "
                f"assembly; got {type(self.metadata).__name__}"
            )
        return self.metadata

    # -- request plumbing ------------------------------------------------------

    @property
    def _platform(self) -> str:
        return str(self._cua_metadata().platform)

    def _effective_system_prompt(self) -> str | None:
        parts = [p for p in (self.system_prompt, self.system_prompt_suffix) if p]
        return "\n\n".join(parts) if parts else None

    def _extra_tool_names(self) -> frozenset[str]:
        assert self.metadata is not None
        return frozenset(tool_schema_name(schema) for schema in self.metadata.extra_tool_schemas)

    def _extra_function_tools(self) -> list[dict[str, Any]]:
        """Env extra tools as native ``FunctionDeclaration`` objects."""
        assert self.metadata is not None
        declarations: list[dict[str, Any]] = []
        for schema in self.metadata.extra_tool_schemas:
            function = schema.get("function", schema)
            declaration: dict[str, Any] = {"name": function["name"]}
            if function.get("description"):
                declaration["description"] = function["description"]
            if function.get("parameters"):
                declaration["parameters"] = copy.deepcopy(function["parameters"])
            declarations.append(declaration)
        return declarations

    def _build_tools(self) -> list[dict[str, Any]]:
        """The native ``tools`` array for this request."""
        metadata = self._cua_metadata()
        tools: list[dict[str, Any]] = []
        if metadata.valid_actions != []:
            computer_use: dict[str, Any] = {
                "environment": self.action_space.environment_for_platform(platform=self._platform)
            }
            excluded = self.action_space.excluded_predefined_functions(
                platform=self._platform,
                valid_actions=metadata.valid_actions,
            )
            if excluded:
                computer_use["excludedPredefinedFunctions"] = sorted(excluded)
            tools.append({"computerUse": computer_use})
        extras = self._extra_function_tools()
        if extras:
            # Native shape: ONE Tool carrying a list of declarations -- not one
            # dict per function as in the OpenAI wire.
            tools.append({"functionDeclarations": extras})
        return tools

    def _admissible_verbs(self) -> frozenset[str]:
        """Verb names the model may emit this request.

        Cannot be derived from ``tools``: the verbs live behind the opaque
        ``computerUse`` entry, so the action space is the only owner that knows.
        """
        metadata = self._cua_metadata()
        if metadata.valid_actions == []:
            return frozenset()
        return self.action_space.visible_predefined_verbs(
            platform=self._platform, valid_actions=metadata.valid_actions
        )

    def _generation_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {}
        thinking_level = self.api_kwargs.get("thinking_level")
        if thinking_level:
            config["thinkingConfig"] = {"thinkingLevel": str(thinking_level).upper()}
        for wire_key, kwarg in (
            ("maxOutputTokens", "max_output_tokens"),
            ("temperature", "temperature"),
        ):
            if self.api_kwargs.get(kwarg) is not None:
                config[wire_key] = self.api_kwargs[kwarg]
        return config

    def _build_payload(self, contents: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {"contents": contents}
        tools = self._build_tools()
        if tools:
            payload["tools"] = tools
        system_prompt = self._effective_system_prompt()
        if system_prompt:
            # systemInstruction is a Content, not a bare string.
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        generation_config = self._generation_config()
        if generation_config:
            payload["generationConfig"] = generation_config
        return payload

    async def _agenerate_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await acompletion_with_retry(
            agenerate_content,
            max_retries=self.api_retry_max,
            base_delay=self.api_retry_base_delay,
            max_delay=self.api_retry_max_delay,
            log_name=self._RETRY_LOG_NAME,
            model=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            payload=payload,
            timeout=self.api_timeout_s,
        )

    async def _finalize_sample(
        self,
        *,
        hooks: list[SampleHook],
        result: LiteRLSample | None,
        completed: bool,
        env: LiteBaseEnv,
    ) -> None:
        await finalize_lite_sample(
            hooks=hooks, result=result, completed=completed, env=env, logger=logger
        )

    async def predict(self, lite_sample: LiteSample) -> LiteMessage:
        raise NotImplementedError(
            f"{type(self).__name__}.predict is not supported; use sample() for rollouts"
        )


# =============================================================================
# Image-index markers on native Parts (agent-local, like both precedents)
# =============================================================================
#
# These run on the FINAL request payload, after truncation, which is why they
# live here rather than in ``utils/history``.


def _provider_visible_image_indices(contents: list[dict[str, Any]]) -> tuple[int, ...]:
    indices: list[int] = []
    for content in contents:
        for part in content.get("parts") or []:
            _collect_marker(part, indices)
    return tuple(indices)


def _collect_marker(part: dict[str, Any], out: list[int]) -> None:
    value = part.get(PROVIDER_VISIBLE_IMAGE_INDEX_MARKER)
    if isinstance(value, int):
        out.append(value)
    function_response = part.get("functionResponse")
    if function_response:
        for nested in function_response.get("parts") or []:
            _collect_marker(nested, out)


def _strip_markers(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove the local-only marker from every Part.

    ⚠️ Correctness-critical, not hygiene: the native REST surface rejects a
    Part carrying an unknown field. Returns a NEW list so the persistent
    ``contents`` keeps its markers for later turns' image accounting.
    """
    stripped = copy.deepcopy(contents)
    for content in stripped:
        for part in content.get("parts") or []:
            _strip_marker(part)
    return stripped


def _strip_marker(part: dict[str, Any]) -> None:
    part.pop(PROVIDER_VISIBLE_IMAGE_INDEX_MARKER, None)
    function_response = part.get("functionResponse")
    if function_response:
        for nested in function_response.get("parts") or []:
            _strip_marker(nested)


# =============================================================================
# Gemini Desktop / Browser Agent
# =============================================================================


@dataclass
class GeminiDesktopUseAgent(_GeminiBaseAgent, key=r"gemini@(desktop|browser)@use"):
    """Rollout loop against Gemini's native desktop/browser computer-use tool."""

    async def sample(
        self,
        env: LiteBaseEnv,
        max_steps: int = 10_000,
        hooks: list[SampleHook] | None = None,
    ) -> LiteRLSample:
        hooks = hooks or []
        result: LiteRLSample | None = None
        completed = False
        contents: list[dict[str, Any]] = []

        try:
            observation = await env.reset()
            image_bytes = observation.image
            instruction = observation.text
            if not instruction:
                raise RuntimeError("No instruction returned on env.reset()")

            trajectory = LiteSample(metadata=env.metadata)
            steps: list[LiteRLStep] = []
            episode_return = 0.0
            terminated = False
            truncated = False
            next_call_id = 0
            observation_image_index: int | None = None
            resize_target = self.resolution

            for step in range(max_steps):
                # 1. Prepare input: store the frame once, then attach it marked.
                image = None
                current_image_index: int | None = None
                if image_bytes:
                    if observation_image_index is None:
                        image = await asyncio.to_thread(decode_image, image_bytes)
                        trajectory.images.append(image)
                        current_image_index = len(trajectory.images) - 1
                    else:
                        image = trajectory.images[observation_image_index]
                        current_image_index = observation_image_index

                if step == 0:
                    parts: list[dict[str, Any]] = [{"text": instruction}]
                    traj_content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
                    if image_bytes and current_image_index is not None:
                        parts.append(
                            mark_provider_visible_image_index(
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": await image_b64(image_bytes, resize_target),
                                    }
                                },
                                current_image_index,
                            )
                        )
                        traj_content.append({"type": "image", "index": current_image_index})
                    contents.append({"role": "user", "parts": parts})
                    trajectory.messages.append({"role": "user", "content": traj_content})

                # 2. Call provider. Order is a hard constraint: truncate, then
                # harvest markers off what will actually be sent, then strip.
                request_contents = filter_to_n_most_recent_images(
                    contents,
                    self.only_n_most_recent_images,
                    min_removal_threshold=self.image_truncation_threshold,
                )
                image_indices = _provider_visible_image_indices(request_contents)
                payload = self._build_payload(_strip_markers(request_contents))

                response, predict_s = await timed(self._agenerate_with_retry(payload))

                # 3. Parse output.
                parsed = parse_response_with_provenance(
                    response,
                    action_space=self.action_space,
                    admissible_verbs=self._admissible_verbs(),
                    extra_tool_names=self._extra_tool_names(),
                    call_id_start=next_call_id,
                    pass_through_unknown=self._PASS_THROUGH_UNKNOWN,
                )
                lite_message = parsed.message
                model_output_error = pop_model_output_error(lite_message)
                next_call_id += len(lite_message.get("tool_calls", []))

                rl_step = append_lite_rl_step(
                    steps=steps,
                    prompt=json.dumps(strip_images(payload), ensure_ascii=False),
                    image_indices=image_indices,
                    response=json.dumps(response, ensure_ascii=False),
                    finish_reason=parsed.finish_reason,
                    truncation_finish_reasons=_TRUNCATION_FINISH_REASONS,
                )
                trajectory.messages.append(lite_message)
                # Replay the model turn BEFORE stepping the env: the native
                # surface requires every functionCall to be immediately
                # followed by its functionResponse block.
                append_provider_assistant_message(contents, replay_parts=parsed.replay_parts)

                # 4. Step env.
                turn = await execute_lite_turn(
                    env=env,
                    trajectory=trajectory,
                    lite_message=lite_message,
                    model_output_error=model_output_error,
                    rl_step=rl_step,
                    tool_execution_timeout_s=self.tool_execution_timeout_s,
                )
                lite_message = turn.message
                step_result = turn.step_result

                # 5. Record trajectory.
                episode_return, terminated, truncated = await record_lite_env_result(
                    trajectory=trajectory,
                    persisted_actions=turn.persisted_actions,
                    step_result=step_result,
                    rl_step=rl_step,
                    step=step,
                    max_steps=max_steps,
                    episode_return=episode_return,
                    append_terminal_tool_feedback=_append_gemini_terminal_tool_feedback,
                )
                await dispatch_lite_step_hooks(
                    hooks=hooks,
                    step_idx=step,
                    image=image,
                    lite_message=lite_message,
                    rl_step=rl_step,
                    step_result=step_result,
                    hook_actions=turn.hook_actions,
                    timings={"predict": predict_s, "act": turn.act_s},
                )

                if terminated or truncated:
                    break

                # 6. Prepare next turn.
                image_bytes, observation_image_index = await append_gemini_provider_feedback(
                    contents=contents,
                    trajectory=trajectory,
                    step_result=step_result,
                    lite_tool_calls=lite_message.get("tool_calls", []),
                    provider_calls=parsed.provider_calls,
                    resize_target=resize_target,
                )
            else:
                truncated = True
                mark_steps_truncated(steps)

            result = build_lite_rl_sample(
                trajectory=trajectory,
                steps=steps,
                episode_return=episode_return,
                terminated=terminated,
                truncated=truncated,
            )
            completed = True
            return result
        finally:
            await self._finalize_sample(hooks=hooks, result=result, completed=completed, env=env)


# =============================================================================
# Gemini Mobile Agent (Android)
# =============================================================================

_GEMINI_MOBILE_SYSTEM_PROMPT = (
    "You are operating an Android phone to complete a user task.\n\n"
    "Coordinates are NORMALIZED to a 0-999 grid over the screenshot, not "
    "pixels: (0,0) is the top-left corner and (999,999) the bottom-right. "
    "The same numbers work on any screen size.\n\n"
    "Guidelines:\n"
    "- Scroll by swiping before concluding that an element is missing.\n"
    "- Type text with the `type` action; do not tap out characters on the "
    "on-screen keyboard.\n"
    "- Use `open_app` to launch an application rather than hunting for its "
    "icon, when that tool is offered.\n"
    "- Verify the resulting screen before declaring the task complete."
)


@dataclass
class GeminiMobileUseAgent(GeminiDesktopUseAgent, key="gemini@mobile@use"):
    """Rollout loop against Gemini's native MOBILE computer-use action space.

    Reuses the desktop ``sample()`` verbatim. Unlike Claude and GPT -- whose
    mobile paths need their own loop because their desktop leg carries an opaque
    native tool and a different tool-result shape -- this family's two platforms
    differ only in the action space, the ``environment`` value, and the
    unknown-call policy.
    """

    #: Mobile precedent: ``*@mobile`` spaces let a named non-native call through
    #: verbatim (``claude@mobile`` / ``gpt@mobile`` both do), while ``*@desktop``
    #: spaces reject it.
    _PASS_THROUGH_UNKNOWN: ClassVar[bool] = True

    action_space: GeminiMobileActionSpace = field(default_factory=GeminiMobileActionSpace)
    system_prompt: str | None = _GEMINI_MOBILE_SYSTEM_PROMPT


__all__ = [
    "GeminiDesktopUseAgent",
    "GeminiMobileUseAgent",
]
