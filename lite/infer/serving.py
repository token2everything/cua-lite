"""sglang utilities: generate_fn wrappers for Engine and server backends.

Usage::

    from sglang import Engine
    from lite.infer.serving import make_engine_generate_fn

    # Wrap an Engine directly (quick-start / single-GPU)
    engine = Engine(model_path="Qwen/Qwen3-VL-4B-Instruct")
    generate_fn = make_engine_generate_fn(engine, {"max_new_tokens": 1024})

    # Or connect to a running server
    from lite.infer.serving import make_server_generate_fn
    generate_fn = make_server_generate_fn("http://localhost:30000", {"max_new_tokens": 1024})

Every generate fn here returns ``{"response": <text>, "finish_reason": <str>}``.
``finish_reason`` is the provider's own stop signal (``"stop"`` / ``"length"`` /
...), read by ``LiteBaseAgent._predict_with_details`` to tell a truncated turn
from a deliberate finish. It is never defaulted: a missing signal is
indistinguishable from a clean stop, so the backends index it and fail loudly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys

import httpx

from lite.utils.path import project_root

_SERVE_SCRIPT = project_root() / "scripts" / "serve_sglang.py"
log = logging.getLogger(__name__)


def make_engine_generate_fn(engine, sampling_kwargs: dict):
    """Return an async generate function backed by a sglang ``Engine`` instance.

    Args:
        engine: A ``sglang.Engine`` instance.
        sampling_kwargs: Passed as ``sampling_params`` to ``engine.async_generate``.
    """
    async def generate(*, prompt: str, images, sampling_seed: int | None = None, **_) -> dict:
        # See make_server_generate_fn: a per-call seed overrides the frozen
        # sampling_kwargs; None → unchanged (back-compat); a caller-pinned
        # sampling_kwargs["sampling_seed"] wins over the per-call one.
        sp = (
            sampling_kwargs
            if sampling_seed is None or "sampling_seed" in sampling_kwargs
            else {**sampling_kwargs, "sampling_seed": sampling_seed}
        )
        out = await engine.async_generate(
            prompt=prompt,
            sampling_params=sp,
            image_data=images or None,
        )
        return {
            "response": out.get("text", ""),
            "finish_reason": out["meta_info"]["finish_reason"]["type"],
        }

    return generate


def make_server_generate_fn(server_url: str, sampling_kwargs: dict):
    """Return an async generate function that POSTs to a sglang server's /generate endpoint.

    Used by eval/sample scripts to connect to a server started via scripts/serve_sglang.py.

    Args:
        server_url: Base URL of the sglang server, e.g. "http://localhost:30000".
        sampling_kwargs: Passed as ``sampling_params`` in the request payload.
    """
    from lite.utils.image import encode_image_b64

    url = f"{server_url.rstrip('/')}/generate"
    # Shared client for connection pooling; lives for the process lifetime.
    # Not explicitly closed — acceptable for long-running eval/sample scripts.
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(None),
        # High-concurrency rollout can exceed 64 in-flight requests; keep the client
        # pool from becoming an unintended bottleneck in front of sglang.
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=256),
    )

    async def generate(*, prompt: str, images, sampling_seed: int | None = None, **_) -> dict:
        # Per-call ``sampling_seed`` (member-distinct, supplied by the rollout
        # caller) overrides the frozen ``sampling_kwargs`` so collect-mode
        # reproducibility can vary the seed per sample without rebuilding this fn.
        # ``None`` → payload unchanged (back-compat). A caller-PINNED
        # ``sampling_kwargs["sampling_seed"]`` wins over the per-call one — the same
        # "pinned seed wins" escape hatch the training path honors.
        sp = (
            sampling_kwargs
            if sampling_seed is None or "sampling_seed" in sampling_kwargs
            else {**sampling_kwargs, "sampling_seed": sampling_seed}
        )
        payload: dict = {"text": prompt, "sampling_params": sp}
        if images:
            payload["image_data"] = list(await asyncio.gather(
                *[asyncio.to_thread(encode_image_b64, img) for img in images]
            ))
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                body = resp.json()
                return {
                    "response": body.get("text", ""),
                    "finish_reason": body["meta_info"]["finish_reason"]["type"],
                }
            except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                last_exc = e
                log.warning("sglang request failed (attempt %d/3): %s", attempt + 1, e)
                await asyncio.sleep(2 ** attempt)
        raise last_exc

    generate.close = client.aclose  # allow explicit cleanup if needed
    return generate


async def start_server(
    model_id: str,
    model_path: str | None = None,
    engine_kwargs: dict | None = None,
) -> tuple[asyncio.subprocess.Process, str]:
    """Launch scripts/serve_sglang.py with --port 0 (auto-pick) and wait until healthy.

    Args:
        model_id: HuggingFace model ID (key in LOCAL_AGENTS).
        model_path: Override model path (e.g. a local checkpoint).
        engine_kwargs: Override sglang engine kwargs (e.g. ``{"tp_size": 4}``).

    Returns:
        ``(proc, server_url)`` where ``server_url`` is e.g. ``"http://localhost:30001"``.
        Caller is responsible for calling ``await stop_server(proc)`` when done.
    """
    cmd = [sys.executable, str(_SERVE_SCRIPT), "--model-id", model_id, "--port", "0"]
    if model_path is not None:
        cmd.extend(["--model-path", model_path])
    if engine_kwargs is not None:
        cmd.extend(["--engine-kwargs", json.dumps(engine_kwargs)])

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE,
        start_new_session=True,  # own process group so stop_server can kill the tree
    )
    assert proc.stdout is not None

    # Read stdout until we see PORT=<port>
    while True:
        line = await proc.stdout.readline()
        if not line:
            raise RuntimeError("serve_sglang.py exited before printing PORT=")
        text = line.decode().strip()
        log.info("[serve_sglang.py] %s", text)
        if text.startswith("PORT="):
            port = int(text.split("=", 1)[1])
            break

    server_url = f"http://localhost:{port}"
    health_url = f"{server_url}/health"
    log.info("Waiting for sglang server at %s ...", health_url)

    # Drain stdout in background so serve_sglang.py doesn't block on pipe buffer
    async def _drain():
        async for _line in proc.stdout:
            pass
    drain_task = asyncio.create_task(_drain())
    drain_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    async with httpx.AsyncClient(timeout=httpx.Timeout(5)) as hc:
        for _ in range(300):  # poll every 2s, up to 10 minutes
            await asyncio.sleep(2)
            if proc.returncode is not None:
                raise RuntimeError(f"sglang server exited early (code {proc.returncode})")
            try:
                resp = await hc.get(health_url)
                resp.raise_for_status()
                log.info("sglang server is ready at %s", server_url)
                return proc, server_url
            except (httpx.HTTPError, OSError):
                pass
    raise RuntimeError("sglang server did not become ready within 600s")


async def stop_server(
    proc: asyncio.subprocess.Process,
    *,
    generate_fn=None,
    timeout: float = 30.0,
) -> None:
    """Terminate the server process group (serve_sglang.py + sglang child).

    Two ordered steps so sglang's graceful-drain doesn't hang forever:

    1. If ``generate_fn`` is given, ``await generate_fn.close()`` first so the
       httpx client's TCP connections are closed before sglang sees SIGTERM.
       Without this, sglang's drain logic waits for clients that never
       disconnect (httpx is configured with ``timeout=None`` for long
       prefill) and the ``Gracefully exiting... Remaining number of
       requests N`` loop runs forever.
    2. SIGTERM the process group, then ``proc.wait()`` with a hard
       ``timeout`` (default 30s). On timeout, escalate to SIGKILL — handles
       the residual case where sglang ignored the disconnects too.
    """
    if generate_fn is not None and hasattr(generate_fn, "close"):
        try:
            await generate_fn.close()
        except Exception as e:
            log.warning("generate_fn.close() raised: %s", e)

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return  # already exited
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        log.warning("sglang did not exit within %.0fs of SIGTERM; SIGKILL", timeout)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()


# ---------------------------------------------------------------------------
# Inference-entry helpers (shared by lite/infer/cli.py — one home for the
# local-model serving glue the rollout/sample scripts used to each inline)
# ---------------------------------------------------------------------------

#: Default sampling for local models. ``temperature=1.0`` matches slime's RL
#: default (avoids argmax mode-collapse when training data has any modal bias);
#: yaml ``agent_kwargs.sampling_kwargs`` and ``--sampling-kwargs`` override it.
DEFAULT_SAMPLING_KWARGS: dict = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 2048}


def load_processor(model_path: str):
    """Load the HF processor for a local model (one home for every local backend)."""
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


def extract_sampling_kwargs(agent_kwargs: dict, cli_sampling_kwargs: dict | None) -> dict:
    """Resolve sampling params: ``DEFAULT_SAMPLING_KWARGS`` < yaml
    (``agent_kwargs['sampling_kwargs']``) < CLI ``--sampling-kwargs``.

    Pops ``sampling_kwargs`` out of *agent_kwargs* in place — it is a generate_fn
    concern consumed here, not an adapter kwarg.
    """
    return {
        **DEFAULT_SAMPLING_KWARGS,
        **(agent_kwargs.pop("sampling_kwargs", None) or {}),
        **(cli_sampling_kwargs or {}),
    }


def make_hf_generate_fn(model_path: str, processor, sampling_kwargs: dict):
    """In-process ``transformers`` generation (the ``hf`` backend — no server)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    try:
        hf_model = AutoModelForImageTextToText.from_pretrained(
            model_path, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
    except ValueError:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
    hf_model.eval()

    temperature = sampling_kwargs.get("temperature", 1.0)
    max_new_tokens = sampling_kwargs.get("max_new_tokens", 2048)

    @torch.no_grad()
    async def generate(*, prompt: str, images, **_) -> dict:
        inputs = processor(
            text=[prompt], images=images or None, return_tensors="pt", padding=True,
        ).to(hf_model.device)
        gen_kw: dict = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                            pad_token_id=processor.tokenizer.eos_token_id)
        if temperature > 0:
            gen_kw["temperature"] = temperature
        prompt_len = inputs["input_ids"].shape[1]
        out_ids = hf_model.generate(**inputs, **gen_kw)
        new_ids = out_ids[0, prompt_len:]
        text = processor.batch_decode(out_ids[:, prompt_len:],
                                      skip_special_tokens=True)[0]
        # HF stops for exactly two reasons — eos, or the token budget — so reading
        # the last generated id is exact rather than a heuristic. The sglang
        # backends read the provider's own field; the consumer sees the same key.
        eos_id = processor.tokenizer.eos_token_id
        return {
            "response": text,
            "finish_reason": "stop" if int(new_ids[-1]) == eos_id else "length",
        }

    return generate


@contextlib.asynccontextmanager
async def local_model(
    model_id: str,
    *,
    model_path: str | None = None,
    backend: str = "sglang",
    engine_kwargs: dict | None = None,
    server_url: str | None = None,
    sampling_kwargs: dict,
):
    """Yield ``(processor, generate_fn)`` for a local model; tear down on exit.

    Backends:
      * ``"sglang"`` (default): connect to *server_url* (falling back to
        ``$SGLANG_SERVER_URL``), else auto-start ``scripts/serve_sglang.py`` and
        SIGTERM it on exit. This is the one place the start/connect/teardown
        dance lives (the rollout/sample scripts each used to inline it).
      * ``"hf"``: in-process ``transformers`` generation (no server).
    """
    model_path = model_path or model_id
    processor = load_processor(model_path)
    if backend == "hf":
        yield processor, make_hf_generate_fn(model_path, processor, sampling_kwargs)
        return
    if backend != "sglang":
        raise ValueError(f"unknown local backend {backend!r} (expected 'sglang' or 'hf')")

    proc = None
    gen = None
    url = server_url or os.environ.get("SGLANG_SERVER_URL")
    try:
        if not url:
            proc, url = await start_server(
                model_id, model_path=model_path, engine_kwargs=engine_kwargs)
        gen = make_server_generate_fn(url, sampling_kwargs)
        yield processor, gen
    finally:
        if proc is not None:
            await stop_server(proc, generate_fn=gen)
