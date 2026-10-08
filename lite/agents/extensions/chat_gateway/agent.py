"""Use canonical desktop tools with a Chat Completions-compatible gateway.

Construct ``ChatGatewayAgent(model_id=..., api_base=..., api_key=...,
request_params={...}, metadata=env.metadata)`` and await ``agent.sample(env)``.
The real model ID is sent unchanged. No factory registration is performed.

The inherited AdapterBasedAgent owns environment stepping, feedback, hooks, and
cleanup. This extension owns HTTP transport and the Chat screenshot-history window.
``request_params`` are HTTP JSON parameters; an SDK-style ``extra_body`` object
is flattened into the outgoing body without changing its values.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from PIL import Image

from lite.agents.core.action_space.base import assemble_tool_schemas
from lite.agents.core.adapter import AsIsAdapter
from lite.agents.core.agent.base import AdapterBasedAgent
from lite.agents.core.agent.utils.request_logs import redact, write_log
from lite.agents.core.agent.utils.retry import retry_delay
from lite.agents.extensions.chat_gateway.prompt import DESKTOP_SYSTEM_PROMPT
from lite.agents.types import GenerateFn, PredictResult
from lite.core import LiteCUAMetadata, LiteMessage, LiteRLSample, LiteRLStep, LiteSample
from lite.core.messages.content import keep_model_visible_content
from lite.core.samples import STATUS_COMPLETED, STATUS_TRUNCATED
from lite.core.tools.calls import validate_lite_tool_call
from lite.utils.image import encode_image_b64

logger = logging.getLogger(__name__)


class ChatGatewayError(RuntimeError):
    """Sanitized transport/protocol failure; raw provider details stay in logs."""


class ChatGatewayProtocolError(ChatGatewayError):
    """A successful completion cannot be projected into canonical model output."""


@dataclass
class ChatGatewayAgent(AdapterBasedAgent):
    """Stateless desktop Chat agent with canonical function-tool schemas.

    ``sample`` reuses AsIsAdapter's image and tool-result handling. Prediction
    projects canonical messages directly, without its T=1 renderer or text
    parser. Requests keep all text, reasoning and paired tool calls/results.
    Only screenshots are limited to ``history_turns`` past screenshots plus the
    current screenshot (default 4); None keeps every screenshot. Screenshots
    exceeding that budget are omitted oldest-first in ``image_fold_size`` groups
    (default 1); groups larger than one leave text placeholders. The stored
    trajectory stays complete. Canonical coordinates remain normalized to [0, 1000].

    Chat tool results contain text; their screenshots become an associated user
    image message after the contiguous tool results. Canonical history still
    stores these observations under role=tool, with their original call IDs.
    ``request_image_dir`` optionally shares logged PNGs with a trajectory image
    directory; it never changes images sent to the model.

    GUI instructions are stored at the start of each sampled trajectory and
    retained in every request. Standalone prediction supplies the same default
    when the caller has not supplied its own system message.
    """

    generate_fn: GenerateFn = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    processor: Any = field(default=None, init=False, repr=False)
    adapter: AsIsAdapter = field(init=False, repr=False)
    preserve_raw_response: bool = field(default=True, init=False, repr=False)
    model_id: str = ""
    api_key: str | None = field(default=None, repr=False)
    api_base: str | None = field(default=None, repr=False)
    metadata: LiteCUAMetadata | None = None
    request_params: dict[str, Any] = field(default_factory=dict)
    request_log_dir: Path | None = None
    request_image_dir: Path | None = None
    history_turns: int | None = 4
    image_fold_size: int = 1
    request_timeout: float | httpx.Timeout = 600.0
    api_retry_max: int = 24
    api_retry_base_delay: float = 2.0
    api_retry_max_delay: float = 60.0
    _request_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (self.model_id, self.api_key, self.api_base)
        ):
            raise ValueError("Chat gateway requires model_id, api_key, and api_base")
        parsed = urlsplit(self.api_base)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "api_base must be an HTTP(S) URL without credentials or query parameters"
            )
        path = parsed.path.rstrip("/")
        full_endpoint = path.endswith("/chat/completions")
        if full_endpoint:
            path = path.removesuffix("/chat/completions")
        elif path.endswith(("/responses", "/messages")):
            raise ValueError("api_base must address the Chat Completions API")
        if not path and not full_endpoint:
            path = "/v1"
        self.api_base = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
        timeouts = (
            self.request_timeout.as_dict().values()
            if isinstance(self.request_timeout, httpx.Timeout)
            else (self.request_timeout,)
        )
        if any(value is None or not math.isfinite(value) or value <= 0 for value in timeouts):
            raise ValueError("request_timeout phases must be finite and positive")
        if self.api_retry_max < 0:
            raise ValueError("api_retry_max must be nonnegative")
        if self.history_turns is not None and (
            type(self.history_turns) is not int or self.history_turns < 0
        ):
            raise ValueError("history_turns must be a nonnegative integer or None")
        if (
            type(self.image_fold_size) is not int
            or self.image_fold_size <= 0
            or (
                self.history_turns is not None
                and self.image_fold_size > self.history_turns + 1
            )
        ):
            raise ValueError(
                "image_fold_size must be a positive integer no greater than the image budget"
            )
        if not isinstance(self.request_params, dict):
            raise TypeError("request_params must contain an HTTP JSON object")
        self.request_params = copy.deepcopy(self.request_params)
        extra_body = self.request_params.pop("extra_body", {})
        if not isinstance(extra_body, dict):
            raise ValueError("request_params.extra_body must contain an HTTP JSON object")
        # This is the SDK's body overlay: explicit false/null values survive.
        self.request_params.update(extra_body)
        if {
            "model", "messages", "tools", "api_key", "api_base", "base_url",
        }.intersection(self.request_params):
            raise ValueError(
                "request_params cannot override model, messages, tools, or connection fields"
            )
        if self.request_params.get("stream", False) is not False:
            raise ValueError("Chat gateway requires stream=False")
        if self.request_params.get("n", 1) != 1:
            raise ValueError("Chat gateway requires a single completion (n=1)")
        if self.metadata is None:
            self.metadata = LiteCUAMetadata(dims=("desktop", "use"))
        if self.metadata.platform not in {"desktop", "browser"}:
            raise ValueError("Chat gateway supports desktop/browser GUI action metadata")
        self.adapter = AsIsAdapter(metadata=self.metadata)
        if self.request_log_dir is not None:
            self.request_log_dir = Path(self.request_log_dir)
        if self.request_image_dir is not None:
            self.request_image_dir = Path(self.request_image_dir)

    async def _append_initial_observation(
        self,
        lite_sample: LiteSample,
        lite_rl_sample: LiteRLSample,
        *,
        image_bytes: bytes | None,
        instruction: str,
        metadata: dict[str, Any] | None,
    ) -> tuple[Image.Image | None, int | None]:
        # Store instructions before the first hook/model call so live JSONL and
        # wire requests describe the same initial conversation.
        lite_sample.messages.append({
            "role": "system",
            "content": [{"type": "text", "text": DESKTOP_SYSTEM_PROMPT}],
        })
        return await super()._append_initial_observation(
            lite_sample, lite_rl_sample, image_bytes=image_bytes,
            instruction=instruction, metadata=metadata,
        )

    def _build_tools(self, metadata: LiteCUAMetadata | None = None) -> list[dict[str, Any]]:
        """Build Chat tool declarations for requests and trajectory metadata."""
        metadata = self.metadata if metadata is None else metadata
        return assemble_tool_schemas(
            self.adapter.action_space,
            self.adapter.action_space.get_tool_schemas(),
            valid_actions=metadata.valid_actions,
            extra_tool_schemas=metadata.extra_tool_schemas,
        )

    def _build_request_kwargs(
        self,
        lite_sample: LiteSample,
        processed_images: list[Image.Image | None],
    ) -> tuple[dict[str, Any], tuple[int, ...]]:
        metadata = lite_sample.metadata
        if not isinstance(metadata, LiteCUAMetadata) or metadata.platform not in {
            "desktop", "browser",
        }:
            raise ValueError("Chat gateway requires desktop/browser GUI metadata")
        messages: list[dict[str, Any]] = []
        image_indices: list[int] = []
        pending_images: list[dict[str, Any]] = []
        source = keep_model_visible_content(lite_sample.messages)
        if not any(message["role"] == "system" for message in source):
            source = [{
                "role": "system",
                "content": [{"type": "text", "text": DESKTOP_SYSTEM_PROMPT}],
            }, *source]
        images_to_skip = 0
        if self.history_turns is not None:
            # Count individual screenshots, including multiple images per tool
            # batch. Earlier text and call/result pairs remain in their original order.
            total_images = sum(
                part["type"] == "image"
                for message in source
                if isinstance(message.get("content"), list)
                for part in message["content"]
            )
            excess_images = max(0, total_images - self.history_turns - 1)
            images_to_skip = (
                (excess_images + self.image_fold_size - 1) // self.image_fold_size
            ) * self.image_fold_size
        # Select the request view before encoding images. Persistent messages,
        # image indices, and the full viewer trajectory are never rewritten.
        for message in source:
            role = message["role"]
            if role != "tool" and pending_images:
                messages.append({"role": "user", "content": pending_images})
                pending_images = []
            content = message.get("content") or []
            parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
            wire_parts: list[dict[str, Any]] = []
            for part in parts:
                if part["type"] == "image":
                    if images_to_skip:
                        images_to_skip -= 1
                        if self.image_fold_size > 1:
                            wire_parts.append({
                                "type": "text",
                                "text": (
                                    f"[Screenshot {part['index']} folded from image history; "
                                    "the original remains in the full trajectory.]"
                                ),
                            })
                        continue
                    index = part["index"]
                    image = processed_images[index]
                    if image is None:
                        raise ValueError("Model-visible image has not been processed")
                    wire_parts.append({
                        "type": "image_url", "image_url": {"url": encode_image_b64(image)},
                    })
                    image_indices.append(index)
                else:
                    wire_parts.append({"type": "text", "text": part["text"]})
            if role == "tool":
                call_id = message["tool_call_id"]
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "\n".join(p["text"] for p in wire_parts if p["type"] == "text"),
                })
                images = [p for p in wire_parts if p["type"] == "image_url"]
                if images:
                    pending_images.append({
                        "type": "text", "text": f"Screenshots from tool_call_id={call_id}:",
                    })
                    pending_images.extend(images)
                continue
            wire: dict[str, Any] = {"role": role, "content": wire_parts}
            if role == "assistant":
                wire["content"] = "\n".join(p["text"] for p in wire_parts) or None
                if "reasoning_content" in message:
                    wire["reasoning_content"] = message["reasoning_content"]
                if message.get("tool_calls"):
                    wire["tool_calls"] = [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["function"]["name"],
                                "arguments": json.dumps(
                                    call["function"]["arguments"],
                                    ensure_ascii=False,
                                    allow_nan=False,
                                ),
                            },
                        }
                        for call in message["tool_calls"]
                    ]
            messages.append(wire)
        if pending_images:
            messages.append({"role": "user", "content": pending_images})
        tools = self._build_tools(metadata)
        body = copy.deepcopy(self.request_params)
        body.update(model=self.model_id, messages=messages, stream=False)
        if tools:
            body["tools"] = tools
        return body, tuple(image_indices)

    async def _acompletion_with_retry(self, **api_kwargs: Any) -> dict[str, Any]:
        self._request_count += 1
        prefix = f"{self._request_count:05d}"
        secrets = (self.api_key or "", self.api_base or "")
        # ASCII escapes avoid widening the entire base64-heavy Python string
        # when the task or reasoning contains CJK text or emoji.
        payload = json.dumps(
            api_kwargs, ensure_ascii=True, allow_nan=False, separators=(",", ":")
        ).encode("ascii")
        if self.request_log_dir is not None:
            await asyncio.to_thread(
                write_log, self.request_log_dir, f"{prefix}.request.json", api_kwargs, secrets,
                image_directory=self.request_image_dir,
            )
        attempts: list[dict[str, Any]] = []
        started = time.monotonic()
        async with httpx.AsyncClient(
            timeout=self.request_timeout, trust_env=False, follow_redirects=False
        ) as client:
            for attempt in range(int(self.api_retry_max) + 1):
                response = None
                attempt_started = time.monotonic()
                error_type = None
                try:
                    response = await client.post(
                        f"{self.api_base}/chat/completions",
                        content=payload,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                except httpx.TransportError as error:
                    error_type = type(error).__name__
                    message = f"Chat gateway transport failed ({type(error).__name__})"
                    retryable = True
                # Time only the provider request, excluding artifact serialization.
                elapsed = time.monotonic() - attempt_started
                attempts.append({
                    "attempt": attempt + 1,
                    "elapsed_seconds": elapsed,
                    "status_code": response.status_code if response is not None else None,
                    "error_type": error_type,
                })
                if self.request_log_dir is not None:
                    await asyncio.to_thread(
                        write_log,
                        self.request_log_dir,
                        f"{prefix}.metadata.json",
                        {
                            "elapsed_seconds": sum(item["elapsed_seconds"] for item in attempts),
                            "wall_seconds": time.monotonic() - started,
                            "attempts": attempts,
                        },
                        secrets,
                    )
                if response is not None:
                    try:
                        data = response.json()
                    except ValueError:
                        data = {"raw_text": response.text}
                    if self.request_log_dir is not None:
                        response_name = (
                            f"{prefix}.response.json"
                            if 200 <= response.status_code < 300
                            else f"{prefix}.attempt-{attempt + 1:02d}.response.json"
                        )
                        await asyncio.to_thread(
                            write_log,
                            self.request_log_dir,
                            response_name,
                            data,
                            secrets,
                            image_directory=self.request_image_dir,
                        )
                    if 200 <= response.status_code < 300:
                        if (
                            not isinstance(data, dict)
                            or data.get("error")
                            or not isinstance(data.get("choices"), list)
                            or len(data["choices"]) != 1
                        ):
                            raise ChatGatewayError(
                                "Chat gateway returned an invalid completion body"
                            )
                        return data
                    message = f"Chat gateway returned HTTP {response.status_code}"
                    retryable = response.status_code in {408, 429, 500, 502, 503, 504}
                if self.request_log_dir is not None:
                    await asyncio.to_thread(
                        write_log,
                        self.request_log_dir,
                        f"{prefix}.attempt-{attempt + 1:02d}.error.json",
                        {
                            "error": message,
                            "status_code": response.status_code if response else None,
                        },
                        secrets,
                        image_directory=self.request_image_dir,
                    )
                if not retryable or attempt >= int(self.api_retry_max):
                    raise ChatGatewayError(message) from None
                delay = retry_delay(
                    attempt,
                    base_delay=float(self.api_retry_base_delay),
                    max_delay=float(self.api_retry_max_delay),
                )
                logger.warning("%s; retrying in %.2fs", message, delay)
                await asyncio.sleep(delay)
        raise AssertionError("unreachable retry state")

    async def _predict_with_details(
        self,
        lite_sample: LiteSample,
        *,
        processed_images: list[Image.Image | None],
        call_id_start: int = 0,
    ) -> PredictResult:
        # Provider IDs, rather than the generic text-parser counter, own Chat
        # result pairing. The inherited sample loop accepts these IDs intact.
        del call_id_start
        body, image_indices = await asyncio.to_thread(
            self._build_request_kwargs, lite_sample, processed_images
        )
        response = await self._acompletion_with_retry(**body)
        choice = response["choices"][0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ChatGatewayProtocolError("Chat gateway omitted the assistant message")
        message = choice["message"]
        if message.get("role") != "assistant" or message.get("function_call"):
            raise ChatGatewayProtocolError("Chat gateway returned an unsupported assistant message")
        if message.get("refusal") or choice.get("finish_reason") == "content_filter":
            raise ChatGatewayProtocolError("Chat gateway refused the model request")
        if choice.get("finish_reason") not in {
            "stop", "tool_calls", "length", "max_tokens", "context_length_exceeded",
        }:
            raise ChatGatewayProtocolError(
                "Chat gateway returned an unsupported completion finish reason"
            )
        content = message.get("content")
        if content is None:
            parts = []
        elif isinstance(content, str):
            parts = [{"type": "text", "text": content}] if content else []
        elif isinstance(content, list) and all(
            isinstance(part, dict) and part.get("type") == "text"
            and isinstance(part.get("text"), str) for part in content
        ):
            parts = copy.deepcopy(content)
        else:
            raise ChatGatewayProtocolError("Chat gateway returned unsupported assistant content")
        canonical: LiteMessage = {"role": "assistant", "content": parts}
        reasoning = message.get("reasoning_content")
        if reasoning is not None:
            if not isinstance(reasoning, str):
                raise ChatGatewayProtocolError("Chat gateway returned non-text reasoning_content")
            canonical["reasoning_content"] = reasoning
        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list):
            raise ChatGatewayProtocolError("Chat gateway returned invalid tool_calls")
        seen_ids = {
            call["id"] for turn in lite_sample.messages for call in turn.get("tool_calls", [])
        }
        calls = []
        for raw_call in raw_calls:
            if (
                not isinstance(raw_call, dict)
                or raw_call.get("type") != "function"
                or not isinstance(raw_call.get("id"), str)
                or not raw_call["id"]
                or not isinstance(raw_call.get("function"), dict)
            ):
                raise ChatGatewayProtocolError("Chat gateway returned an invalid function call")
            function = raw_call["function"]
            try:
                arguments = json.loads(function["arguments"])
            except (KeyError, TypeError, ValueError):
                raise ChatGatewayProtocolError(
                    "Chat gateway returned invalid function arguments"
                ) from None
            if not isinstance(arguments, dict):
                raise ChatGatewayProtocolError("Chat gateway function arguments must be an object")
            call = {
                "id": raw_call["id"],
                "type": "function",
                "function": {"name": function.get("name"), "arguments": arguments},
            }
            if (
                validate_lite_tool_call(call, "Chat tool call", require_id=True) is not None
                or call["id"] in seen_ids
            ):
                raise ChatGatewayProtocolError(
                    "Chat gateway returned invalid or duplicate tool call IDs"
                )
            seen_ids.add(call["id"])
            calls.append(call)
        if calls:
            canonical["tool_calls"] = calls
        elif choice.get("finish_reason") == "tool_calls" or not any(p["text"] for p in parts):
            raise ChatGatewayProtocolError("Chat gateway omitted usable assistant output")

        # RL/debug prompts retain the actual Chat envelope but reference the
        # sample's image slots. Full wire images are archived by request_logs.
        prompt_body = copy.deepcopy(body)
        referenced = iter(image_indices)
        for turn in prompt_body["messages"]:
            if isinstance(turn.get("content"), list):
                for part in turn["content"]:
                    if part["type"] == "image_url":
                        part["image_url"]["url"] = f"image:{next(referenced)}"
        secrets = (self.api_key or "", self.api_base or "")
        status = (
            STATUS_TRUNCATED
            if choice.get("finish_reason") in {"length", "max_tokens", "context_length_exceeded"}
            else STATUS_COMPLETED
        )
        return PredictResult(
            lite_message=canonical,
            agent_message=copy.deepcopy(message),
            step=LiteRLStep(
                prompt=json.dumps(redact(prompt_body, secrets), ensure_ascii=False),
                image_indices=image_indices,
                response=json.dumps(redact(message, secrets), ensure_ascii=False),
                status=status,
            ),
        )
