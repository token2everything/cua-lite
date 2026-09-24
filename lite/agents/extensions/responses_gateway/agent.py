"""Run native computer tools through a stateless Responses-compatible gateway.

Construct ``ResponsesGatewayAgent(model_id=..., api_base=..., api_key=...,
request_params={"reasoning": {"context": "all_turns", "effort": "max"}},
metadata=env.metadata)`` and await ``agent.sample(env)``. The inherited GPT
runtime owns canonical actions, environment feedback, hooks, and cleanup; this
extension owns HTTP transport and the gateway's full-history wire contract.

``request_params`` contains HTTP JSON parameters, rather than LiteLLM kwargs.
The real model ID is sent unchanged. No factory registration is performed.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit, urlunsplit

import httpx

from lite.agents.core.agent.utils.request_logs import write_log
from lite.agents.core.agent.utils.retry import retry_delay
from lite.agents.models.gpt.agent import GPTDesktopUseAgent
from lite.agents.models.gpt.utils.parse import GPTParsedOutput

logger = logging.getLogger(__name__)


def _reasoning_texts(item: dict[str, Any]) -> list[str]:
    """Read native reasoning text, keeping provider whitespace and block order."""
    content = item.get("content")
    if not isinstance(content, list):
        return []
    return [
        part["text"]
        for part in content
        if isinstance(part, dict)
        and part.get("type") == "reasoning_text"
        and isinstance(part.get("text"), str)
        and part["text"].strip()
    ]


class ResponsesGatewayError(RuntimeError):
    """A sanitized transport or protocol failure, with raw details in request logs."""


@dataclass
class ResponsesGatewayAgent(GPTDesktopUseAgent):
    """Desktop Responses agent with explicit, client-managed gateway history.

    ``api_base`` accepts an API base or a full ``/responses`` endpoint.
    ``request_log_dir`` stores each request, response, and failed HTTP attempt;
    data-URL images are stored once under ``images/`` or ``request_image_dir``
    when sharing the trajectory's image directory. Request timeouts and
    cancellations never replay environment actions or alter model inputs.
    """

    request_params: dict[str, Any] = field(default_factory=dict)
    request_log_dir: Path | None = None
    request_image_dir: Path | None = None
    request_timeout: float = 600.0
    api_key: str | None = field(default=None, repr=False)
    api_base: str | None = field(default=None, repr=False)
    _request_count: int = field(default=0, init=False, repr=False)
    _API_KWARGS_DEFAULTS: ClassVar[dict[str, Any]] = {
        "chain_previous_response": False,
        "parallel_tool_calls": False,
        "truncation": None,
    }

    def __post_init__(self) -> None:
        self.api_kwargs = copy.deepcopy(self.api_kwargs or {})
        self.api_kwargs.update(chain_previous_response=False, truncation=None)
        self.full_history_size = None
        super().__post_init__()
        if not all(
            isinstance(value, str) and value.strip()
            for value in (self.model_id, self.api_key, self.api_base)
        ):
            raise ValueError("Responses gateway requires model_id, api_key, and api_base")
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
        full_endpoint = path.endswith("/responses")
        if full_endpoint:
            path = path.removesuffix("/responses")
        elif path.endswith(("/chat/completions", "/messages")):
            raise ValueError("api_base must address the Responses API")
        if not path and not full_endpoint:
            path = "/v1"
        self.api_base = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
        if not math.isfinite(self.request_timeout) or self.request_timeout <= 0:
            raise ValueError("request_timeout must be finite and positive")
        if self.api_retry_max < 0:
            raise ValueError("api_retry_max must be nonnegative")
        if not isinstance(self.request_params, dict):
            raise TypeError("request_params must contain an HTTP JSON object")
        self.request_params = copy.deepcopy(self.request_params)
        owned = {
            "model",
            "input",
            "tools",
            "previous_response_id",
            "api_key",
            "api_base",
            "base_url",
        }
        if owned.intersection(self.request_params):
            raise ValueError(
                "request_params cannot override model, input, tools, or connection fields"
            )
        if self.request_params.get("store", False) is not False:
            raise ValueError("Responses gateway requires store=False")
        if self.request_params.get("stream", False) is not False:
            raise ValueError("Responses gateway requires stream=False")
        if self.request_params.get("truncation") not in (None, "disabled"):
            raise ValueError("Responses gateway does not allow automatic history truncation")
        include = self.request_params.get("include", [])
        if not isinstance(include, list) or not all(isinstance(item, str) for item in include):
            raise ValueError("request_params.include must be a list of strings")
        if self.request_log_dir is not None:
            self.request_log_dir = Path(self.request_log_dir)
        if self.request_image_dir is not None:
            self.request_image_dir = Path(self.request_image_dir)

    def _build_request_kwargs(
        self,
        *,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        previous_response_id: str | None,
        instructions: str | None = None,
        prompt_cache_key: str | None = None,
    ) -> dict[str, Any]:
        del previous_response_id, prompt_cache_key
        body = copy.deepcopy(self.request_params)
        body.update(
            model=self.model_id,
            input=input_items,
            tools=tools or [],
            store=False,
            stream=False,
        )
        body.setdefault("parallel_tool_calls", False)
        include = body.setdefault("include", [])
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        if body.get("truncation") is None:
            body.pop("truncation", None)
        if instructions:
            body["instructions"] = instructions
        return body

    async def _aresponses_with_retry(self, **api_kwargs: Any) -> Any:
        self._request_count += 1
        prefix = f"{self._request_count:05d}"
        secrets = (self.api_key or "", self.api_base or "")
        # ASCII escapes keep the base64-heavy serialized body at one byte per
        # character even when its text contains CJK characters or emoji.
        payload = json.dumps(
            api_kwargs, ensure_ascii=True, allow_nan=False, separators=(",", ":")
        ).encode("ascii")
        if self.request_log_dir is not None:
            await asyncio.to_thread(
                write_log, self.request_log_dir, f"{prefix}.request.json", api_kwargs, secrets,
                image_directory=self.request_image_dir,
            )
        async with httpx.AsyncClient(
            timeout=self.request_timeout, trust_env=False, follow_redirects=False
        ) as client:
            for attempt in range(int(self.api_retry_max) + 1):
                response = None
                retryable = False
                try:
                    response = await client.post(
                        f"{self.api_base}/responses",
                        content=payload,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                except httpx.TransportError as error:
                    message = f"Responses gateway transport failed ({type(error).__name__})"
                    retryable = True
                else:
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
                            or not isinstance(data.get("output"), list)
                            or not data["output"]
                        ):
                            raise ResponsesGatewayError(
                                "Responses gateway returned an invalid output body"
                            )
                        if data.get("error") or data.get("status") not in (
                            "completed",
                            "incomplete",
                        ):
                            raise ResponsesGatewayError(
                                "Responses gateway returned an unsuccessful response"
                            )
                        if not isinstance(data.get("id"), str) or not data["id"]:
                            raise ResponsesGatewayError("Responses gateway omitted the response ID")
                        for item in data["output"]:
                            if not isinstance(item, dict) or item.get("type") not in (
                                "reasoning",
                                "message",
                                "computer_call",
                                "function_call",
                            ):
                                raise ResponsesGatewayError(
                                    "Responses gateway returned an unsupported output item"
                                )
                            if item.get("type") == "reasoning":
                                encrypted = item.get("encrypted_content")
                                if not (
                                    isinstance(encrypted, str) and encrypted.strip()
                                ) and not _reasoning_texts(item):
                                    raise ResponsesGatewayError(
                                        "Responses gateway omitted replayable reasoning; "
                                        "expected encrypted_content or reasoning_text"
                                    )
                        return data
                    message = f"Responses gateway returned HTTP {response.status_code}"
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
                    raise ResponsesGatewayError(message) from None
                delay = retry_delay(
                    attempt,
                    base_delay=float(self.api_retry_base_delay),
                    max_delay=float(self.api_retry_max_delay),
                )
                logger.warning("%s; retrying in %.2fs", message, delay)
                await asyncio.sleep(delay)
        raise AssertionError("unreachable retry state")

    def _parse_output_items(
        self,
        output_items: list[dict[str, Any]],
        *,
        resolution: tuple[int, int],
        active_provider_tool_names: frozenset[str] | None = None,
        call_id_start: int = 0,
    ) -> GPTParsedOutput:
        # The gateway can emit either published Responses computer-call shape.
        # Normalize only the parser view, then restore raw provenance so the
        # inherited history builder replays the original item without mutation.
        normalized = []
        originals = {}
        reasoning_parts = []
        for item in output_items:
            projected = item
            if item.get("type") == "reasoning":
                reasoning_parts.extend(_reasoning_texts(item))
            elif item.get("type") == "computer_call":
                actions = item.get("actions")
                if not actions and isinstance(item.get("action"), dict):
                    actions = [item["action"]]
                    projected = {**item, "actions": actions}
                if (
                    not isinstance(actions, list)
                    or not actions
                    or not all(isinstance(action, dict) for action in actions)
                ):
                    raise ResponsesGatewayError("Responses computer_call has no valid action list")
            normalized.append(projected)
            originals[id(projected)] = item
        parsed = super()._parse_output_items(
            normalized,
            resolution=resolution,
            active_provider_tool_names=active_provider_tool_names,
            call_id_start=call_id_start,
        )
        message = parsed.message
        if reasoning_parts:
            # Full native reasoning has its own canonical channel. Provider
            # summaries remain inline_reasoning content owned by the GPT parser.
            message = {**message, "reasoning_content": "\n".join(reasoning_parts)}
        return replace(
            parsed,
            message=message,
            provider_calls=tuple(
                replace(call, item=originals[id(call.item)]) for call in parsed.provider_calls
            ),
        )
