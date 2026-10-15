"""Binary wire frames for /reset and /step responses.

Keeps the per-turn screenshot OUT of the JSON body, so neither the env-server
loop nor the client loop ever ``json.dumps``/``json.loads`` the multi-MB blob.
Only a tiny scalar header is JSON (``text``/``metadata``/``info`` inline);
observation and tool-result images ride raw in trailing length-delimited
sections with no JSON over the blob.

Frame layout (big-endian lengths)::

    MAGIC(4) | header_len(4) | header_json(header_len)
      | [reset image bytes OR step result image bytes...]

Reset ``header`` = {v, kind="reset", metadata, text, shot_n}, where
``shot_n`` is either ``[]`` or one raw-image byte length.
Step ``header`` = {v, kind="step", reward, terminated, truncated, info,
results}. Each result header is {tool_call_id, text, metadata, error, shot_n},
where ``shot_n`` can carry one length per captured result image.
For both reset and step results, ``[]`` means no images.
``metadata``/``info`` are env-owned JSON passed through verbatim; this module
does not read or rewrite any env key (see ``LiteEnvObservation.metadata`` for
the "keep it small" contract that keeps the header cheap).
Lossless: ``decode(encode(x)) == x`` for every field, ``None`` preserved.

The MAGIC is bumped on every wire change (``LEF1`` -> ``LEF2`` -> ``LEF3`` ->
``LEF4`` -> ``LEF5`` -> ``LEF6``), so a version skew surfaces as
:class:`BadMagic` at byte 0. Decode
errors are TYPED so the transport gate maps terminal-vs-retryable without
guessing: :class:`BadMagic`/:class:`BadVersion` are terminal ("upgrade"),
:class:`TruncatedFrame` is retryable. The typed contract is TOTAL — every
truncation site (empty / short / cut-inside-header / short tail) raises
:class:`TruncatedFrame`, independent of any caller gate ordering.

Round-trip test:  uv run pytest tests/gym/remote/test_frame.py
"""
from __future__ import annotations

import json
import math

from lite.core.tools.results import LiteToolResult
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult

FRAME_MAGIC = "LEF6"
_MAGIC = FRAME_MAGIC.encode("ascii")
FRAME_VERSION = 6
_HEAD = 8  # MAGIC(4) + header_len(4)


class FrameError(Exception):
    """Base for step-frame decode errors."""


class BadMagic(FrameError):
    """Wrong/foreign MAGIC — a version skew or corrupt body. Terminal."""


class BadVersion(FrameError):
    """MAGIC ok but header ``v`` mismatches FRAME_VERSION. Terminal."""


class TruncatedFrame(FrameError):
    """Buffer shorter than the frame declares (empty / short / cut tail). Retryable."""


def _guard_mapping_serializable(header: dict, name: str) -> None:
    """Raise ``TypeError`` naming the offending mapping key.

    ``json.dumps`` itself names nothing, so we walk the top-level keys and
    ``json.dumps(v)`` each — this recurses, so a nested/list offender is caught
    too (localized to its top-level key). Reset ``metadata`` is ``dict | None``;
    step ``info`` is always a ``dict``.
    """
    d = header[name]
    if d is None:
        return
    for k, v in d.items():
        try:
            json.dumps(v, allow_nan=False)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"frame header {name}[{k!r}] is not JSON-serializable ({e})"
            ) from e


def _guard_reset_serializable(header: dict) -> None:
    _guard_mapping_serializable(header, "metadata")


def _guard_step_serializable(header: dict) -> None:
    _guard_mapping_serializable(header, "info")
    for i, result in enumerate(header["results"]):
        metadata = result["metadata"]
        if metadata is None:
            continue
        for k, v in metadata.items():
            try:
                json.dumps(v, allow_nan=False)
            except (TypeError, ValueError) as e:
                raise TypeError(
                    "frame header "
                    f"results[{i}].metadata[{k!r}] is not JSON-serializable ({e})"
                ) from e


def _pack_frame(header: dict, tail_parts: list[bytes]) -> bytes:
    # allow_nan=False: a NaN/Inf reward (a non-RFC token, and NaN != NaN voids the
    # bit-identity claim) fails fast here rather than shipping.
    hb = json.dumps(header, allow_nan=False).encode("utf-8")
    out = bytearray()
    out += _MAGIC
    out += len(hb).to_bytes(4, "big")
    out += hb
    for part in tail_parts:
        out += part
    return bytes(out)


def encode_reset_observation(observation: LiteEnvObservation) -> bytes:
    """Serialize a reset observation to the binary frame."""
    shot = observation.image  # raw PNG bytes or None — already the wire form
    header = {
        "v": FRAME_VERSION,
        "kind": "reset",
        "metadata": observation.metadata,
        "text": observation.text,  # str | None — JSON string in the header (Scope)
        "shot_n": [] if shot is None else [len(shot)],  # [] sentinel
    }
    _guard_reset_serializable(header)
    return _pack_frame(header, [] if shot is None else [shot])


def encode_step_result(result: LiteEnvStepResult) -> bytes:
    """Serialize a step result to the binary frame (no json over the blob)."""
    result_images: list[bytes] = []
    result_headers = []
    for tool_result in result.results:
        result_headers.append({
            "tool_call_id": tool_result.tool_call_id,
            "text": tool_result.text,
            "metadata": tool_result.metadata,
            "error": tool_result.error,
            "shot_n": [len(image) for image in tool_result.images],
        })
        result_images.extend(tool_result.images)
    header = {
        "v": FRAME_VERSION,
        "kind": "step",
        "reward": result.reward,
        "terminated": result.terminated,
        "truncated": result.truncated,
        "info": result.info,
        "results": result_headers,
    }
    _guard_step_serializable(header)
    return _pack_frame(header, result_images)


def _reject_nonfinite(tok: str) -> float:
    """``json.loads`` ``parse_constant`` hook: reject ``NaN``/``Infinity`` tokens.

    Symmetric with the ``allow_nan=False`` encode guard — a conforming peer never
    emits these, so a non-finite token means a malformed/foreign frame. Raising
    ``ValueError`` routes it to the retryable ``TruncatedFrame`` path (a clean
    re-fetch recovers) rather than leaking a silent NaN reward downstream.
    """
    raise ValueError(f"non-finite json constant {tok!r}")


def _parse_finite_float(tok: str) -> float:
    """``json.loads`` ``parse_float`` hook for RFC-valid overflow literals."""
    value = float(tok)
    if not math.isfinite(value):
        raise ValueError(f"non-finite json number {tok!r}")
    return value


def _decode_header(buf: bytes | bytearray) -> tuple[dict[str, object], memoryview, int, int]:
    n = len(buf)
    if n < _HEAD:
        raise TruncatedFrame(f"buffer {n} < {_HEAD}-byte frame head")
    mv = memoryview(buf)
    if mv[:4].tobytes() != _MAGIC:
        raise BadMagic("bad step-frame magic")
    hlen = int.from_bytes(mv[4:8], "big")
    end_hdr = _HEAD + hlen
    if n < end_hdr:  # cut inside the header — must NOT reach json.loads
        raise TruncatedFrame(f"buffer {n} < header end {end_hdr}")
    try:
        header = json.loads(
            mv[_HEAD:end_hdr].tobytes().decode("utf-8"),
            parse_constant=_reject_nonfinite,  # symmetric with encode's allow_nan=False
            parse_float=_parse_finite_float,
        )
    except ValueError as e:  # JSONDecodeError ⊂ ValueError; also catches _reject_nonfinite
        raise TruncatedFrame(f"header not valid json ({e})") from e
    if not isinstance(header, dict):
        raise BadVersion("frame header must be an object")
    if header.get("v") != FRAME_VERSION:
        raise BadVersion(f"frame v={header.get('v')!r} != {FRAME_VERSION}")
    return header, mv, end_hdr, n


def _expect_kind(header: dict[str, object], kind: str) -> None:
    if header.get("kind") != kind:
        raise BadVersion(f"frame kind={header.get('kind')!r} != {kind!r}")


def _read_single_image(
    *,
    shot_lengths: object,
    mv: memoryview,
    off: int,
    n: int,
    label: str,
) -> tuple[bytes | None, int]:
    if not isinstance(shot_lengths, list):
        raise BadVersion(f"{label}.shot_n must be a list")
    if len(shot_lengths) > 1:
        raise BadVersion(
            f"{label}.shot_n has {len(shot_lengths)} images, expected <= 1"
        )
    if not shot_lengths:
        return None, off

    shot_n = shot_lengths[0]
    if (
        not isinstance(shot_n, int)
        or isinstance(shot_n, bool)
        or shot_n < 0
    ):
        raise BadVersion(f"{label}.shot_n[0] must be a non-negative integer")
    end_shot = off + shot_n
    if n < end_shot:
        raise TruncatedFrame(f"buffer {n} < {label} image end {end_shot}")
    return mv[off:end_shot].tobytes(), end_shot


def _read_images(
    *,
    shot_lengths: object,
    mv: memoryview,
    off: int,
    n: int,
    label: str,
) -> tuple[list[bytes], int]:
    if not isinstance(shot_lengths, list):
        raise BadVersion(f"{label}.shot_n must be a list")
    images: list[bytes] = []
    for idx, shot_n in enumerate(shot_lengths):
        if (
            not isinstance(shot_n, int)
            or isinstance(shot_n, bool)
            or shot_n < 0
        ):
            raise BadVersion(f"{label}.shot_n[{idx}] must be a non-negative integer")
        end_shot = off + shot_n
        if n < end_shot:
            raise TruncatedFrame(f"buffer {n} < {label} image {idx} end {end_shot}")
        images.append(mv[off:end_shot].tobytes())
        off = end_shot
    return images, off


def _require_results(header: dict[str, object]) -> list[object]:
    results = header.get("results")
    if not isinstance(results, list):
        raise BadVersion("step.results must be a list")
    return results


def _require_result_header(result_header: object, index: int) -> dict[str, object]:
    if not isinstance(result_header, dict):
        raise BadVersion(f"result[{index}] must be an object")
    if "call_id" in result_header:
        raise BadVersion(f"result[{index}] must use tool_call_id, not call_id")
    if "tool_call_id" not in result_header:
        raise BadVersion(f"result[{index}].tool_call_id is required")
    tool_result_id = result_header["tool_call_id"]
    if tool_result_id is not None and not isinstance(tool_result_id, str):
        raise BadVersion(f"result[{index}].tool_call_id must be a string or null")
    return result_header


def decode_reset_observation(buf: bytes | bytearray) -> LiteEnvObservation:
    """Inverse of :func:`encode_reset_observation`."""
    header, mv, off, n = _decode_header(buf)
    _expect_kind(header, "reset")
    image, off = _read_single_image(
        shot_lengths=header.get("shot_n"),
        mv=mv,
        off=off,
        n=n,
        label="reset",
    )
    if off != n:
        raise TruncatedFrame(f"trailing bytes: consumed {off} of {n}")
    return LiteEnvObservation(
        image=image, text=header["text"], metadata=header["metadata"]
    )


def decode_step_result(buf: bytes | bytearray) -> LiteEnvStepResult:
    """Inverse of :func:`encode_step_result`. Only the small header is json-parsed."""
    header, mv, off, n = _decode_header(buf)
    _expect_kind(header, "step")
    results: list[LiteToolResult] = []
    for i, raw_result_header in enumerate(_require_results(header)):
        result_header = _require_result_header(raw_result_header, i)
        images, off = _read_images(
            shot_lengths=result_header.get("shot_n"),
            mv=mv,
            off=off,
            n=n,
            label=f"result[{i}]",
        )
        results.append(
            LiteToolResult(
                tool_call_id=result_header["tool_call_id"],
                images=images,
                text=result_header["text"],
                metadata=result_header["metadata"],
                error=result_header.get("error"),
            )
        )
    if off != n:
        raise TruncatedFrame(f"trailing bytes: consumed {off} of {n}")

    return LiteEnvStepResult(
        reward=header["reward"],
        terminated=header["terminated"],
        truncated=header["truncated"],
        info=header["info"],
        results=results,
    )
