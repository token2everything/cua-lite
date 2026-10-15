"""GPT image sizing and processed-dimension helpers.

Owns everything about the pixel frame a Responses request sends and the frame
the API reports back: the client-side resize applied before the call, the
``input_items`` lookup that reads the processed dimensions, and the request
wrapper that pairs the two. Desktop normalizes coordinates by the processed
frame; mobile/grounding declare a fixed sent frame, so a mismatch is an error
there. Response records, failure surface, and usage telemetry live in
``lite.agents.models.gpt.utils.responses``.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from typing import Any

import httpx
from PIL import Image

from lite.agents.models.gpt.utils.responses import _normalized_gpt_response

logger = logging.getLogger(__name__)

# Image detail for screenshot inputs. Always ``original`` per OpenAI's
# computer-use guide: it preserves full resolution (up to 10.24M px) and
# improves click accuracy; "high"/"low" are discouraged for computer use. When
# the token budget needs it, downscale client-side via ``_resize_for_api`` (and
# remap coords in the action-space layer) rather than lowering detail.
_DEFAULT_IMAGE_DETAIL = "original"

# Responses API content blocks whose ``image_url`` carries a base64 payload.
# Used by processed-dimension lookup here, and by the history module for log
# redaction and history truncation.
_IMAGE_BLOCK_TYPES = {"input_image", "computer_screenshot"}


def _resize_for_api(
    png: bytes,
    target: tuple[int, int] | None,
) -> tuple[str, int, int]:
    """Return the base64 PNG payload and sent-frame dimensions.

    When ``target`` is set, the image is stretched exactly to that size.
    Coordinate normalization uses processed dimensions resolved after the API
    call.

    ``obs.image`` is not guaranteed PNG — mobilegym emits JPEG — while every
    caller labels the payload ``data:image/png``. The no-resize path therefore
    re-encodes anything else instead of forwarding the source bytes under a
    wrong media type. This path is the DEFAULT (``target`` is ``None`` whenever
    no ``resolution`` is configured), so a mislabel here would affect every
    frame of every turn.
    """
    img = Image.open(io.BytesIO(png))
    w, h = img.size
    if target is None or (w, h) == target:
        if img.format == "PNG":
            return base64.b64encode(png).decode("utf-8"), w, h
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8"), w, h
    img = img.resize(target, Image.LANCZOS)
    logger.debug("Stretch resize: %dx%d -> %dx%d", w, h, target[0], target[1])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), target[0], target[1]


# GPT may internally resize images. Desktop coordinates are normalized by the
# processed dimensions the API reports; mobile/grounding declare sent-frame
# pixels, so a processed-dimension mismatch is a contract error there.
async def _fetch_processed_image_dims(
    response_id: str,
    model_id: str,
    api_base: str | None,
    api_key: str | None,
) -> list[tuple[int, int]]:
    """GET /responses/{id}/input_items?include[]=message.input_image.image_url

    Return processed-image dimensions for each input_image in request order.
    Empty list means the response has no input_image. Uses the same
    OpenAI-compatible base URL configured for litellm; the base must include
    ``/v1``.
    """
    del model_id  # image-item lookup is routed by OPENAI_BASE_URL.
    base = (api_base or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip(
        "/"
    )
    key = api_key or os.environ.get("OPENAI_API_KEY") or ""
    url = f"{base}/responses/{response_id}/input_items?include[]=message.input_image.image_url"
    headers = {"Authorization": f"Bearer {key}"}

    async with httpx.AsyncClient(timeout=30) as cli:
        r = await cli.get(url, headers=headers)
    r.raise_for_status()
    data = r.json()

    dims: list[tuple[int, int]] = []

    def _walk(o: Any) -> None:
        if isinstance(o, dict):
            if o.get("type") in _IMAGE_BLOCK_TYPES:
                iu = o.get("image_url")
                if isinstance(iu, str) and iu.startswith("data:image"):
                    m = re.match(r"data:image/[\w+]+;base64,(.+)", iu)
                    if m:
                        raw = base64.b64decode(m.group(1))
                        im = Image.open(io.BytesIO(raw))
                        dims.append(im.size)
            for v in o.values():
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)

    _walk(data)
    return dims


def _unwrap_litellm_response_id(rid: str) -> str:
    """litellm wraps the upstream response_id like
    ``resp_<base64(litellm:...response_id:resp_REAL...)>``. Unwrap to the
    real Azure/OpenAI response_id so ``GET /responses/{id}/...`` works.
    """
    if not rid.startswith("resp_"):
        return rid
    payload = rid[len("resp_") :]
    try:
        # litellm uses unpadded b64 sometimes; add padding before decoding.
        decoded = base64.b64decode(payload + "=" * (-len(payload) % 4)).decode(
            "utf-8", errors="ignore"
        )
    except Exception:  # noqa: BLE001
        return rid
    m = re.search(r"response_id:(resp_[A-Za-z0-9]+)", decoded)
    return m.group(1) if m else rid


async def _call_api_with_actual_dim(
    call_fn: Any,
    api_kwargs: dict[str, Any],
    *,
    sent_w: int | None,
    sent_h: int | None,
    model_id: str,
    api_base: str | None,
    api_key: str | None,
) -> tuple[Any, int | None, int | None]:
    """Call API, then resolve the actual processed-image dims for the
    current turn's screenshot.

    Returns ``(response, actual_w, actual_h)``:
        * ``actual_w/h``: dims the API ECHOES BACK for the most recent image
          block (= dims the model actually saw, after any silent server-side
          resize). Use these for coord normalization.
        * Falls back to ``(sent_w, sent_h)`` when this request sent no image
          block, and when the processed-dimension lookup does not answer.

    Logs ``WARNING`` when sent != processed so silent DS is observable, and
    when the lookup degrades to the sent frame.
    """

    def request_contains_image_block(value: Any) -> bool:
        if isinstance(value, dict):
            if value.get("type") in _IMAGE_BLOCK_TYPES:
                return True
            return any(request_contains_image_block(item) for item in value.values())
        if isinstance(value, list):
            return any(request_contains_image_block(item) for item in value)
        return False

    response = await call_fn(**api_kwargs)
    if sent_w is None or sent_h is None:
        return response, sent_w, sent_h
    if not request_contains_image_block(api_kwargs.get("input")):
        return response, sent_w, sent_h
    normalized = _normalized_gpt_response(response)
    if normalized.status == "failed" or normalized.error:
        # No coordinates will be parsed from a failed provider response; keep
        # the response-failure owner as the loud error path.
        return response, sent_w, sent_h
    # The lookup is best-effort: when it does not answer, degrade to the sent
    # frame and warn. Degrading is safe because the sent dims ARE the processed
    # dims unless the API resized, and the resize case is not silent — the
    # callers own a separate, loud guard ("GPT API auto-downsampled ...") that
    # fires whenever processed dims are known and differ from sent. So an
    # unanswered lookup is not an unguarded coordinate hazard; raising here
    # instead turns an intermittent provider gap into certain trajectory loss.
    # It is also not plausible on the fixed-frame surfaces: a 1080x2400 mobile
    # screenshot is ~2.6 MP, far under the API's image budget (the only
    # observed auto-downsample was a 6016x3384 grounding frame).
    rid = normalized.response_id
    if not rid:
        logger.warning(
            "GPT response carried no id (model=%s); falling back to sent dims "
            "%dx%d for coord normalization",
            model_id,
            sent_w,
            sent_h,
        )
        return response, sent_w, sent_h
    rid = _unwrap_litellm_response_id(rid)
    try:
        processed = await _fetch_processed_image_dims(
            rid,
            model_id,
            api_base,
            api_key,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Could not fetch processed image dims (%s); falling back to sent "
            "dims %dx%d for coord normalization",
            e,
            sent_w,
            sent_h,
        )
        return response, sent_w, sent_h
    if not processed:
        logger.warning(
            "Processed image dim lookup returned no image items; falling back to "
            "sent dims %dx%d for coord normalization",
            sent_w,
            sent_h,
        )
        return response, sent_w, sent_h
    actual_w, actual_h = processed[-1]
    if (actual_w, actual_h) != (sent_w, sent_h):
        logger.warning(
            "API auto-downsampled image: sent %dx%d → processed %dx%d (model=%s). "
            "Desktop normalizes by the processed frame; mobile/grounding (which declare "
            "a fixed sent frame) treat this as an error.",
            sent_w,
            sent_h,
            actual_w,
            actual_h,
            model_id,
        )
    return response, actual_w, actual_h
