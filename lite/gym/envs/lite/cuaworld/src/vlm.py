"""VLM-judge shim for gym-anything verifiers.

gym-anything's VLM-judged ``verifier.py`` files do
``from gym_anything.vlm import sample_trajectory_frames, query_vlm,
get_final_screenshot``. We don't vendor the upstream package, so ``adapter``
registers THIS module as ``gym_anything.vlm`` in ``sys.modules`` before loading a
verifier. It mirrors the upstream contract:

* ``query_vlm(prompt, images=[paths]) -> {success, response, parsed, error}`` —
  a real call to the VLM via ``litellm.completion``. The ordinary path returns a
  dict-compatible envelope that also proxies string methods to ``response``,
  and falls back to ``parsed`` for ``.get("schema_key")`` / subscript reads.
  Several pinned verifiers treat the return as either text
  (``.strip()`` / ``.upper()`` / ``.lower()``) or a parsed dict; both input forms must
  keep working. Provider and image failures raise ``VLMProviderError`` instead of
  returning a score-like false result. Asked for JSON (``return_json=True`` and
  friends, see ``_JSON_RETURN_KWARGS``), it
  returns the PARSED OBJECT instead of that envelope, because those verifiers
  subscript the result directly.
* ``sample_trajectory_frames`` / ``get_final_screenshot`` read host image PATHS
  from the ``traj`` dict (``adapter`` populates ``frames`` / ``final_screenshot``
  from the complete episode trajectory).

Configuration — the default judge MODEL ID lives in the env config loaded by this
process at import time (normally ``configs/default.yaml`` under
``env_kwargs.judge.model``; a whole-file ``LITE_CUAWORLD_CONFIG`` replacement must
carry the same key). Rollout YAML ``env_kwargs`` are not passed into verifier
``query_vlm`` calls; use ``VLM_MODEL`` / ``LITE_CUAWORLD_VLM_MODEL`` for per-run
judge overrides. litellm infers the provider from the id and resolves that
provider's credentials and base URL from the environment itself, exactly like the
gpt agent-rollout path (``--model-id`` + ``OPENAI_API_KEY`` plus optional
``OPENAI_BASE_URL``; see ``lite/agents/models/gpt/agent.py``).
There is no backend switch, no per-provider key table, and no model-id prefix
patching::

    # Default route: nothing VLM-specific.
    export OPENAI_API_KEY=...
    # Optional custom endpoint.
    export OPENAI_BASE_URL=<custom endpoint>

    # A different judge model for one run.
    export VLM_MODEL=<id>

    # A non-OpenAI provider: export ITS standard env vars and prefix the id.
    export ANTHROPIC_API_KEY=...   VLM_MODEL=anthropic/claude-sonnet-4-5
    export GEMINI_API_KEY=...      VLM_MODEL=gemini/gemini-2.5-pro

    # Your own server: name the model AND point at it.
    export VLM_MODEL=openai/Qwen/Qwen3-VL-8B-Instruct
    export VLM_BASE_URL=<custom endpoint>

``VLM_BASE_URL`` / ``VLM_API_KEY`` are OPTIONAL per-judge overrides; leave them
unset and litellm's own environment resolution wins (see ``get_vlm_config``).
``VLM_MODEL`` also falls back to ``LITE_CUAWORLD_VLM_MODEL``. The historical
``VLM_MAX_RETRIES`` key is kept for compatibility but means total provider
attempts in this shim; ``VLM_TIMEOUT`` is per attempt.

The judge runs host-side in whichever process evaluates — export these in the
env-server's environment, not the container's. Smoke-test the resolved judge
without a rollout::

    uv run python -c "from lite.gym.envs.lite.cuaworld.src.vlm import redacted_vlm_config, query_vlm; \
print(redacted_vlm_config()); print(query_vlm('Reply with the word OK.'))"
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from lite.gym.utils import config as env_config

logger = logging.getLogger(__name__)
_SECRET_ENV_KEYS = (
    "OPENAI_API_KEY",
    "VLM_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
)

#: The judge model, from the whole env config loaded in this process
#: (``env_kwargs.judge.model``) — a declared default, not a literal buried here,
#: so it is reviewable next to the rest of the env's knobs. litellm infers the
#: provider from the id and resolves that provider's credentials and base URL from
#: the environment itself, the same contract the agent side runs on
#: (`lite/agents/models/gpt/agent.py` builds `{"model": self.model_id, …}` and
#: attaches api_key/api_base only when explicitly set). See the module docstring.
_DEFAULT_MODEL: str = str(
    env_config.load(str(Path(__file__).resolve().parents[1])).env_kwargs["judge"]["model"]
)


def _config_value(
    overrides: dict[str, Any] | None,
    direct_keys: str | tuple[str, ...],
    env_key: str,
    default: Any = None,
) -> Any:
    keys = (direct_keys,) if isinstance(direct_keys, str) else direct_keys
    if overrides is not None:
        for key in (*keys, env_key):
            if key in overrides:
                return overrides[key]
    return os.environ.get(env_key, default)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _redact_text(text: str) -> str:
    redacted = text
    for key in _SECRET_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            redacted = redacted.replace(value, "<redacted>")
    return re.sub(
        r"(?i)(api[_-]?key|authorization|bearer)\s*[:=]\s*['\"]?[^'\"\s,}]+",
        r"\1=<redacted>",
        redacted,
    )


def get_vlm_config(
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the judge call from a model id; litellm does the rest.

    Everything a provider needs — credentials, base URL, request shape — litellm
    already reads from the environment once it knows the model. So this returns a
    model plus two OPTIONAL overrides, and never invents a value: an unset
    ``base_url``/``api_key`` is ``None``, which the caller omits from the request
    so litellm's own resolution wins. (Passing an empty string instead would
    OVERRIDE that resolution with a broken value.)
    """
    model = _optional_text(
        _config_value(
            overrides,
            "model",
            "VLM_MODEL",
            os.environ.get("LITE_CUAWORLD_VLM_MODEL", _DEFAULT_MODEL),
        )
    )
    if model is None:
        raise ValueError(
            "VLM model is empty; set env_kwargs.judge.model, VLM_MODEL, "
            "or LITE_CUAWORLD_VLM_MODEL"
        )
    base_url = _optional_text(
        _config_value(overrides, ("base_url", "api_base"), "VLM_BASE_URL")
    )
    api_key = _optional_text(_config_value(overrides, "api_key", "VLM_API_KEY"))
    if api_key is None and base_url:
        # Self-hosted (vLLM/SGLang/llama.cpp) endpoints ignore the key, but litellm's
        # OpenAI path rejects api_key=None CLIENT-SIDE — "The api_key client option
        # must be set" — before a request is ever made. So the documented recipe
        # (`VLM_MODEL=openai/Qwen/… VLM_BASE_URL=...`, no key)
        # fails 100% of the time without this. A placeholder is what the OpenAI SDK
        # expects here and what these servers are built to ignore.
        api_key = "EMPTY"
    raw_retries = _config_value(overrides, "max_retries", "VLM_MAX_RETRIES", 3)
    try:
        max_attempts = int(raw_retries)
    except (TypeError, ValueError) as exc:
        raise ValueError("VLM_MAX_RETRIES must be a positive integer") from exc
    if max_attempts < 1:
        raise ValueError("VLM_MAX_RETRIES must be a positive integer")
    # A FLOOR, not an option. Without a timeout a stalled endpoint hangs the verifier
    # thread — and with it the enclosing /step — indefinitely. Malformed values fail
    # loudly; otherwise verifier runs can silently outlive their episode budget.
    timeout_value = _config_value(overrides, "timeout", "VLM_TIMEOUT")
    try:
        timeout = float(timeout_value) if timeout_value is not None else 180.0
    except (TypeError, ValueError) as exc:
        raise ValueError("VLM_TIMEOUT must be a positive finite number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("VLM_TIMEOUT must be a positive finite number")
    return {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        # Compatibility key name. Value semantics are total attempts, not
        # extra retries; adapter timeout budgeting relies on this.
        "max_retries": max_attempts,
        "timeout": timeout,
    }


def redacted_vlm_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the resolved judge config with credentials masked for smoke tests."""
    config = dict(get_vlm_config(overrides))
    if config.get("api_key"):
        config["api_key"] = "<redacted>"
    return config


def validated_image_mime(image_bytes: bytes) -> str:
    """Return a supported MIME type only for a fully decodable image."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
            image_format = image.format
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError("invalid or empty VLM image") from exc
    if image_format == "PNG":
        return "image/png"
    if image_format == "JPEG":
        return "image/jpeg"
    if image_format == "GIF":
        return "image/gif"
    if image_format == "WEBP":
        return "image/webp"
    raise RuntimeError(f"unsupported VLM image format: {image_format}")


def _upstream_image_payload(path: str) -> tuple[str, str] | None:
    """Encode an image only after proving it is a decodable screenshot."""
    try:
        image_path = Path(path)
        if not image_path.exists():
            logger.warning("Image not found: %s", path)
            return None
        image_bytes = image_path.read_bytes()
        mime = validated_image_mime(image_bytes)
        encoded = base64.b64encode(image_bytes).decode("utf-8")
    except Exception as exc:  # noqa: BLE001 - mirror upstream best-effort encoding
        logger.warning("Error encoding image %s: %s", path, exc)
        return None
    if not encoded:
        return None
    return encoded, mime


#: Sentinel for "the response carried no JSON at all", distinct from a parsed `{}`.
_NOT_JSON = object()


def _structured_json(text: str) -> Any:
    """Parse the shapes a real model actually emits, or return ``_NOT_JSON``.

    Bare JSON, JSON inside ```json fences, and JSON with prose either side. Split
    out of ``parse_vlm_json`` so the ``return_json`` path can tell a genuine object
    from that function's yes/no KEYWORD fallback — the fallback's ``{"answer": …}``
    is truthy and would pass a caller's "did the judge respond?" test while
    answering None to every schema key it asked for.
    """
    if not text:
        return _NOT_JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```" in text:
        try:
            block = text.split("```json")[-1] if "```json" in text else text.split("```")[1]
            return json.loads(block.split("```")[0].strip())
        except (json.JSONDecodeError, IndexError):
            pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            return {"items": json.loads(m.group())}
        except json.JSONDecodeError:
            pass
    return _NOT_JSON


def parse_vlm_json(text: str) -> dict[str, Any]:
    """Mirror gym-anything's tolerant VLM response parser."""
    structured = _structured_json(text)
    if structured is not _NOT_JSON:
        return structured
    if not text:
        return {}
    low = text.lower()
    parsed: dict[str, Any] = {}
    if "yes" in low and "no" not in low:
        parsed["answer"] = True
    elif "no" in low and "yes" not in low:
        parsed["answer"] = False
    elif "true" in low:
        parsed["answer"] = True
    elif "false" in low:
        parsed["answer"] = False
    if "high confidence" in low or "confident" in low:
        parsed["confidence"] = "high"
    elif "medium confidence" in low or "moderate" in low:
        parsed["confidence"] = "medium"
    elif "low confidence" in low or "uncertain" in low:
        parsed["confidence"] = "low"
    return parsed


class VLMResponse(dict[str, Any]):
    """Dict envelope that remains usable by text-style upstream call sites."""

    def _response_text(self) -> str:
        return str(dict.get(self, "response", "") or "")

    def _parsed(self) -> dict[str, Any]:
        parsed = dict.get(self, "parsed", {})
        return parsed if isinstance(parsed, dict) else {}

    def get(self, key: Any, default: Any = None) -> Any:
        if dict.__contains__(self, key):
            return dict.get(self, key, default)
        if isinstance(key, str):
            return self._parsed().get(key, default)
        return default

    def __getitem__(self, key: str) -> Any:
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        parsed = self._parsed()
        if key in parsed:
            return parsed[key]
        raise KeyError(key)

    def __str__(self) -> str:
        return self._response_text()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self._response_text() == other
        return dict.__eq__(self, other)

    def __contains__(self, key: object) -> bool:
        if dict.__contains__(self, key):
            return True
        if isinstance(key, str) and key in self._parsed():
            return True
        return isinstance(key, str) and key in self._response_text()

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._response_text(), name, None)
        if attr is None:
            raise AttributeError(name)
        return attr


def _envelope(
    *,
    success: bool,
    response: str,
    parsed: dict[str, Any],
    error: str,
) -> VLMResponse:
    return VLMResponse({
        "success": success,
        "response": response,
        "parsed": parsed,
        "error": error,
    })


class VLMProviderError(BaseException):
    """The judge could not be reached / did not answer at all.

    This deliberately subclasses ``BaseException`` rather than ``Exception``.
    Pinned verifiers often wrap judge calls in broad ``except Exception`` blocks
    and then return a normal score. Provider/image failures are infrastructure
    failures, so they must bypass those task-local handlers and be surfaced by the
    CUA-Lite verifier worker as invalid samples, not reward 0.
    """


_PROVIDER_FAILURES: list[str] = []


def clear_vlm_provider_failures() -> None:
    _PROVIDER_FAILURES.clear()


def pop_vlm_provider_failures() -> list[str]:
    failures = list(_PROVIDER_FAILURES)
    _PROVIDER_FAILURES.clear()
    return failures


def _raise_provider_error(message: str, *, cause: BaseException | None = None) -> None:
    _PROVIDER_FAILURES.append(message)
    if cause is None:
        raise VLMProviderError(message)
    raise VLMProviderError(message) from cause


def _success_result(text: str) -> VLMResponse:
    return _envelope(
        success=True,
        response=text,
        parsed=parse_vlm_json(text),
        error="",
    )


#: Kwargs by which a pinned verifier asks for the PARSED JSON OBJECT back rather than
#: this shim's envelope. Their prompts spell out a JSON schema and their scoring reads
#: the schema keys straight off the return value (`vlm_response.get("has_legend")`),
#: so handing back `{success, response, parsed, error}` silently answers None to every
#: key — a dict, so `isinstance(x, dict)` guards pass and the whole VLM block scores 0.
#:
#: Derived by an exhaustive AST census, not by guesswork — basis: every ``*.py`` under
#: the 3216 on-disk task dirs (2974 files, 6 of which do not ``ast.parse``), giving 920
#: ``query_vlm`` call sites with ZERO ``**kwargs`` splats, so the kwarg names below are
#: the complete set. Per-kwarg site counts: ``return_json`` 3, ``output_schema`` 2,
#: ``json_response`` 1 — plus three kwargs deliberately NOT listed here:
#:
#: * ``options`` (3 sites / 2 tasks: diagrams_net/mobile_app_wireframe_prd,
#:   gcompris/connect_dots_mystery ×2) — a multiple-choice request whose sites want
#:   mutually incompatible types back (a bare str vs a dict keyed "text"), so it stays
#:   absorbed.
#: * ``format_response`` (1 site: gvsig_desktop/convert_vector_to_raster:120) — it
#:   READS ``.get("answer_bool")`` from a prompt that asks a prose yes/no question and
#:   specifies no schema at all. No JSON ever comes back, so JSON mode handed it ``{}``
#:   and ``answer_bool`` was False by construction; the envelope answers False just the
#:   same, and this way the site stops being a JSON site — which matters, because it is
#:   the ONE unguarded site that used to be in this tuple (see ``_json_result``).
#: * ``model`` (1 site: freecad/create_bent_bracket:111) — silently absorbed, and that
#:   is ACCEPTABLE: honouring it would let one upstream verifier hard-pin its own model
#:   and override the deployment's configured judge, which is exactly the per-call
#:   backend switch this shim refuses to grow. Its branch is score-neutral anyway: it
#:   tests ``parsed["answer"] == "yes"`` while ``parse_vlm_json`` only ever sets a
#:   BOOLEAN there, so both arms of the ``if`` add 0 points.
_JSON_RETURN_KWARGS = (
    "return_json",
    "json_response",
    "output_schema",
)


def _json_result(text: str) -> dict[str, Any]:
    """The parsed object a ``return_json``-style caller expects, for MODEL TEXT only.

    This handles exactly one of the two ways the judge can let a caller down, and it
    matters that they are handled differently:

    * Unparseable model TEXT (here) — the judge answered, we just cannot read a JSON
      object out of it. Degrade to ``{}``: falsy, so a caller that tests
      ``if vlm_response:`` reports "no response" rather than mistaking a broken judge
      for a confident all-False verdict, and ``.get`` on it still answers None.
    * A PROVIDER failure (``query_vlm``'s outer handler) — no answer at all. That
      raises ``VLMProviderError`` instead, because returning a dict-like value there
      lets verifier-local broad exception handlers or falsey checks downgrade an
      infrastructure outage into a normal reward.
    """
    parsed = _structured_json(text)
    if isinstance(parsed, list):  # a bare top-level array, e.g. `[{...}]`
        parsed = {"items": parsed}
    if not isinstance(parsed, dict):
        logger.warning(
            "lite.cuaworld query_vlm: no JSON object in VLM response: %.200r", text
        )
        return {}
    return parsed


def _retry_delay(attempt: int) -> float:
    """Seconds to sleep before retrying provider attempt ``attempt`` (0-based).

    ``min(60, 2**attempt)`` seconds, scaled by U(0.5, 1.5). Capped exponential
    backoff with jitter: a bare ``2 ** attempt`` made every concurrent verifier
    retry in lockstep against an already-overloaded judge.

    The agent model families run the same formula from
    ``lite.agents.core.agent.utils.retry``. This judge keeps its own copy
    because ``lite/gym`` never imports ``lite/agents``; the two policies are
    free to diverge.
    """
    return min(60.0, 2**attempt) * (0.5 + random.random())


def query_vlm(prompt: str, images: list[str] | dict[str, Any] | None = None,
              image: str | dict[str, Any] | None = None, max_tokens: int = 2048,
              temperature: float = 0.1, top_p: float = 0.95,
              config: dict | None = None, **kwargs: Any) -> dict[str, Any]:
    """Query the VLM with a prompt and optional image file paths.

    ``**kwargs`` absorbs the extra kwargs the pinned verifiers pass. Rejecting them
    raised TypeError inside the verifier, which is a hard 0 on an UNGUARDED site and
    silently drops the whole VLM branch on a guarded one. The census in
    ``_JSON_RETURN_KWARGS`` pins 11 kwarg-passing sites, of which **5 sites across 4
    tasks are unguarded**: `diagrams_net/mobile_app_wireframe_prd:136`,
    `gcompris/connect_dots_mystery:94` and `:121`, `freecad/create_bent_bracket:111`,
    `gvsig_desktop/convert_vector_to_raster:120`.

    Any of ``_JSON_RETURN_KWARGS`` additionally switches the RETURN to the parsed
    object. ``options``, ``format_response`` and ``model`` stay absorbed — see the
    constant for why each one is excluded.

    A PROVIDER or image failure raises ``VLMProviderError`` on every path. Returning
    an envelope there is unsafe: ordinary call sites frequently treat the envelope as
    text, and broad verifier ``except Exception`` blocks can convert the resulting
    error into reward 0.
    """
    # Upstream also calls this POSITIONALLY as `query_vlm(frames, prompt)` (the
    # pre-shim gym_anything signature). Detect the inversion and swap, otherwise
    # `prompt` binds a list — the provider rejects a non-string content part — and
    # `images` binds the prompt string, which `list()` splats into one "path" per
    # CHARACTER (hundreds of "Image not found: e / o / n" warnings, zero frames).
    if not isinstance(prompt, str) and isinstance(images, str):
        prompt, images = images, prompt
    # One pinned verifier passes the complete trajectory dict as `images=traj`.
    # Upstream's helper contract is path lists, so adapt that shape here instead
    # of handing dict keys to Path(...) and silently sending a text-only VLM query.
    if isinstance(images, dict):
        images = sample_trajectory_frames(images, n=5)
    if isinstance(image, dict):
        image_paths = sample_trajectory_frames(image, n=5)
        image = None
    else:
        image_paths = []
    # A bare path where a list is expected: `list("…/a.png")` would splat it too.
    if isinstance(images, str):
        images = [images]
    paths = list(images or [])
    if image_paths:
        paths = image_paths + paths
    if image:
        paths = [image] + paths
    # Tolerate callers (upstream verifiers) that pass a sampled-frames LIST as a single
    # element or via image=; flatten one level so each entry is a real path string.
    # Without this, Path(<list>) raises and the frames never reach the VLM → false 0.
    _flat: list[str] = []
    for _p in paths:
        _flat.extend(_p) if isinstance(_p, (list, tuple)) else _flat.append(_p)
    paths = _flat
    want_json = any(kwargs.get(key) for key in _JSON_RETURN_KWARGS)
    content: list[dict] = []
    for p in paths:
        payload = _upstream_image_payload(p)
        if payload is None:
            continue
        b64, mime = payload
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}})
    if paths and not content:
        message = "no valid VLM images could be encoded"
        logger.warning("lite.cuaworld query_vlm failed: %s", message)
        _raise_provider_error(message)
    content.append({"type": "text", "text": prompt})
    try:
        import litellm
        effective = get_vlm_config(config)
        model = effective["model"]
        max_attempts = int(effective["max_retries"])
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                completion_kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "drop_params": True,
                }
                # Pass credentials ONLY when explicitly configured — the same
                # rule the agent side follows (lite/agents/models/gpt/agent.py
                # `if self.api_key: … if self.api_base: …`). litellm resolves
                # OPENAI_BASE_URL / OPENAI_API_KEY on its own, so the model id
                # is the only thing a caller must supply; forwarding an empty
                # string here would OVERRIDE that resolution with a bad value.
                if effective["base_url"]:
                    completion_kwargs["api_base"] = effective["base_url"]
                if effective["api_key"]:
                    completion_kwargs["api_key"] = effective["api_key"]
                if effective["timeout"] is not None:
                    completion_kwargs["timeout"] = effective["timeout"]
                resp = litellm.completion(
                    **completion_kwargs,
                )
                txt = resp.choices[0].message.content or ""
                if want_json:
                    return _json_result(txt)
                return _success_result(txt)
            except Exception as exc:  # noqa: BLE001 - bounded provider retry
                last_error = exc
                if attempt + 1 < max_attempts:
                    time.sleep(_retry_delay(attempt))
        assert last_error is not None
        raise last_error
    except Exception as exc:  # noqa: BLE001 - convert provider failures uniformly
        message = _redact_text(f"{type(exc).__name__}: {exc}")
        logger.warning("lite.cuaworld query_vlm failed: %s", message)
        # A dead judge must not look like an answer. VLMProviderError subclasses
        # BaseException so task-local `except Exception` blocks cannot downgrade it
        # into a normal score.
        _raise_provider_error(message, cause=exc)


def sample_trajectory_frames(
    traj: dict[str, Any],
    num_samples: int = 3,
    include_first: bool = True,
    include_last: bool = True,
    *,
    n: int | None = None,
) -> list[str]:
    """Sample up to ``num_samples`` frame paths from ``traj['frames']``."""
    if n is not None:
        num_samples = n
    frames = traj.get("frames", [])
    if not frames:
        final = traj.get("final_screenshot")
        return [final] if final else []
    if len(frames) <= num_samples:
        return list(frames)
    indices: list[int] = []
    if include_first:
        indices.append(0)
    if include_last:
        indices.append(len(frames) - 1)
    remaining = num_samples - len(indices)
    if remaining > 0:
        step = (len(frames) - 1) / (remaining + 1)
        for offset in range(1, remaining + 1):
            index = int(offset * step)
            if index not in indices and 0 <= index < len(frames):
                indices.append(index)
    return [frames[index] for index in sorted(set(indices))]


def get_final_screenshot(traj: dict[str, Any]) -> str | None:
    for key in ("post_verification_screenshot", "final_screenshot", "last_frame"):
        p = traj.get(key)
        if p and Path(p).exists():
            return p
    return None


def get_first_screenshot(traj: dict[str, Any]) -> str | None:
    p = traj.get("first_frame")
    if p and Path(p).exists():
        return p
    frames = traj.get("frames", [])
    return frames[0] if frames and Path(frames[0]).exists() else None
