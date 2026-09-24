"""
Image utilities for CUA-Lite.

Usage:
    from lite.utils.image import smart_resize, encode_image_b64

    resized_h, resized_w = smart_resize(height=1080, width=1920)
    data_uri = encode_image_b64(pil_image)
"""

from __future__ import annotations

import base64
import io
import math
import os
from typing import Any

from PIL import Image


def load_images(paths: list[str], *, image_root: str | None = None) -> list[Image.Image]:
    """Load images from file paths at their native size.

    Deliberately does NOT resize. The one resize an exported/rolled-out sample
    gets is the adapter's ``agent_kwargs.resolution`` stage
    (:meth:`BaseAgentAdapter.process_image`), which is the only code that runs
    at BOTH rollout and export — so model-IO parity is structural instead of
    something two configs have to agree on. A resize here would be a third
    resize point rollout does not have.

    Args:
        paths: Image file paths (absolute, or relative if image_root is set).
        image_root: Prepended to relative paths (e.g. CUA_LITE_RAW_DATASETS_ROOT for preproc data).
    """
    images = []
    for p in paths:
        abs_path = os.path.join(image_root, p) if image_root else p
        images.append(Image.open(abs_path).convert("RGB"))
    return images


def decode_hf_images(raw_images: list) -> list[Image.Image | None]:
    """Decode images from parquet to PIL.

    Handles:
      - dict with "bytes" key (HF Image feature via pyarrow)
      - raw bytes (PNG bytes stored directly via write_records_to_parquet)
      - PIL.Image (already decoded)
      - None (sparse processed-image placeholder)
    """
    images = []
    for img in raw_images:
        if img is None:
            images.append(None)
        elif isinstance(img, dict) and "bytes" in img:
            images.append(Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
        elif isinstance(img, bytes):
            images.append(Image.open(io.BytesIO(img)).convert("RGB"))
        elif isinstance(img, Image.Image):
            images.append(img)
        else:
            raise ValueError(f"Unknown image format: {type(img)}")
    return images


def encode_image_b64(img: Image.Image) -> str:
    """Encode a PIL Image as a base64 PNG data URI (e.g. for sglang /generate)."""
    buf = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def encode_png(src: Image.Image | bytes | bytearray) -> bytes:
    """Encode a screenshot to raw PNG **bytes** (the in-memory wire form).

    Accepts either a PIL image (a producer that must PNG-encode) or already-PNG
    bytes (a pass-through producer):

    - ``Image.Image``: saved as PNG to an in-memory buffer.
    - ``bytes``/``bytearray``: already-encoded PNG bytes — returned verbatim (a
      ``bytearray`` is normalized to ``bytes`` so the frame codec's ``+=`` is happy).
    """
    if isinstance(src, Image.Image):
        buf = io.BytesIO()
        src.save(buf, format="PNG")
        return buf.getvalue()
    return bytes(src)


def decode_image(png: bytes) -> Image.Image:
    """Decode raw PNG bytes to a PIL RGB image.

    ``.convert("RGB")`` is load-bearing: palette/L/RGBA screenshots must not
    reach vision processors with a different mode or tensor shape.
    """
    # Detach pixels from the source stream even for RGB PNGs; retaining the
    # lazy Image.open object also retains its compressed BytesIO per turn.
    with io.BytesIO(png) as source, Image.open(source) as img:
        return img.convert("RGB")


def png_from_b64(s: str | None) -> bytes | None:
    """Decode a base64 screenshot string to raw PNG **bytes** (the None-guard baked in).

    Container ``*/docker/*`` servers emit base64 as their own wire protocol; the
    host decodes to bytes once, off the model's b64 currency. ``None`` in →
    ``None`` out (the ``resp.get("screenshot_b64")`` case); ``""`` → ``b""``.
    """
    if s is None:
        return None
    return base64.b64decode(s)


async def png_from_b64_async(s: str | None) -> bytes | None:
    """Off-loop :func:`png_from_b64` — decode a container's base64 without blocking
    the event loop. Use the sync :func:`png_from_b64` when already inside a
    ``to_thread``/executor block (else you double-offload)."""
    import asyncio

    if s is None:
        return None
    return await asyncio.to_thread(base64.b64decode, s)


def normalize_resolution(value: Any) -> tuple[int, int] | None:
    """Coerce an ``agent_kwargs.resolution`` value into the declared ``(W, H)``.

    Every field that holds a resolution is declared ``tuple[int, int]``, and
    every config that sets one spells it as a yaml/parquet **list** which reaches
    the field by plain ``setattr``. Without this the declared type is a lie and
    ``img.size == resolution`` compares a tuple to a list — always ``False``, so
    an identity target-resize never short-circuits. Call it from the owning
    dataclass's ``__post_init__`` so the bad type cannot outlive construction.

    ``None`` means "no target-resize" and passes through. Anything that is not a
    pair raises here, at construction, rather than at the first screenshot.
    """
    if value is None:
        return None
    width, height = value
    return (int(width), int(height))


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
    max_long_side: int = 8192,
) -> tuple[int, int]:
    """Resize dimensions to satisfy vision model constraints.

    Returns (resized_height, resized_width) such that:
    1. Both are divisible by ``factor``
    2. Total pixels is within [min_pixels, max_pixels]
    3. Aspect ratio is approximately preserved

    ``max_long_side`` is a **pre-rounding** downscale trigger, not a guaranteed
    bound on the result. The rescale at ``max(height, width) > max_long_side``
    runs *before* :func:`round_by_factor`, so rounding can push the long side
    back over the cap: ``smart_resize(8193, 41)`` returns a long side of
    ``round_by_factor(8192, 28) == 8204``. The overshoot is always exactly
    ``round_by_factor(max_long_side, factor) - max_long_side`` (+12 px at the
    default ``factor=28``) and only occurs for factors where
    ``max_long_side % factor > factor / 2`` — ``factor=14``/``32``/``56``/``112``
    never violate. It is unreachable in this tree today: no caller overrides
    ``max_long_side``, no env screenshot or dataset image has a long side over
    8192, and near-square inputs are clamped by ``max_pixels`` long before this
    branch matters. Behaviour is pinned as-is by
    ``tests/utils/test_image_resize_geometry.py``; moving the guard after the
    rounding would be a behaviour change needing its own golden re-pin.
    """
    if height < 2 or width < 2:
        raise ValueError(f"height:{height} or width:{width} must be >= 2")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be < 200, got {height}/{width}")

    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)

    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar
