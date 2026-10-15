"""Gemini ``generateContent`` HTTP boundary.

The ONLY module in this family that knows the URL, the auth header, or httpx.
Everything downstream works on the parsed JSON dict.

Why the native REST surface instead of litellm's OpenAI-compatible path:
litellm has no representation for ``computerUse`` at all -- its Gemini tool
mapping knows only googleSearch / googleSearchRetrieval / enterpriseWebSearch /
url_context / code_execution / googleMaps, and anything else is dropped -- so
the tool never reaches the wire, taking ``environment`` with it. On this path
the field is honoured, which is what makes ``gemini@mobile`` a real mobile
action space rather than a desktop one wearing a mobile label.

This module deliberately does NOT normalize or validate the response body.
``resp.json()`` is already a dict -- unlike litellm, which returns either a
pydantic object or a dict and therefore needs a normalizer. Structural
judgement ("not a dict", "no parts") belongs to :mod:`.parse`; duplicating it
here would give one guard two owners.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

#: Only this spelling works. ``/gemini/v1beta/...`` and ``/gemini/models/...``
#: both return 500 through a litellm proxy.
GENERATE_CONTENT_PATH = "/v1beta/models/{model}:generateContent"

#: Gemini authenticates by header, not bearer token.
API_KEY_HEADER = "x-goog-api-key"


class GeminiTransportError(RuntimeError):
    """A ``generateContent`` call did not return 2xx.

    Carries the request shape that most often causes it, because the proxy
    collapses several distinct server-side rejections into an opaque
    ``500 Internal Server Error`` with no field name:

    * an ``excludedPredefinedFunctions`` entry the environment does not offer
      (e.g. ``navigate`` under ``ENVIRONMENT_DESKTOP``);
    * a ``Part`` still carrying the local-only image-index marker;
    * a malformed or missing ``thoughtSignature``.

    Without those two summaries in the message the three are indistinguishable
    in a rollout log.
    """


def _request_shape_summary(payload: dict[str, Any]) -> str:
    """The two payload facts worth having in a transport error message."""
    tools = payload.get("tools") or []
    excluded: list[str] = []
    for tool in tools:
        if isinstance(tool, dict):
            computer_use = tool.get("computerUse") or {}
            excluded.extend(computer_use.get("excludedPredefinedFunctions") or [])
    part_keys: set[str] = set()
    for content in payload.get("contents") or []:
        for part in content.get("parts") or []:
            if isinstance(part, dict):
                part_keys.update(part)
    return f"excluded={sorted(excluded)} part_keys={sorted(part_keys)}"


async def agenerate_content(
    *,
    model: str,
    api_base: str | None,
    api_key: str | None,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """POST one ``generateContent`` request and return ``resp.json()``.

    Retry is the caller's job: wrap this in
    :func:`lite.agents.core.agent.utils.retry.acompletion_with_retry`, which
    retries on any ``Exception``.
    """
    if not api_base:
        raise GeminiTransportError(
            "api_base is required for the Gemini generateContent transport "
            "(proxy root URL); pass it on the agent or set it in the config"
        )
    url = api_base.rstrip("/") + GENERATE_CONTENT_PATH.format(model=model)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers[API_KEY_HEADER] = api_key

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=payload)

    if response.status_code // 100 != 2:
        raise GeminiTransportError(
            f"generateContent {response.status_code} for model={model!r}: "
            f"{response.text[:500]} | {_request_shape_summary(payload)}"
        )

    try:
        return response.json()
    except json.JSONDecodeError as exc:  # 2xx with a non-JSON body is a bug
        raise GeminiTransportError(
            f"generateContent returned {response.status_code} with a non-JSON "
            f"body: {response.text[:200]!r}"
        ) from exc


__all__ = [
    "API_KEY_HEADER",
    "GENERATE_CONTENT_PATH",
    "GeminiTransportError",
    "agenerate_content",
]
