"""Trainer-side hook to materialize cua-lite's lazy multimodal payloads.

Called once per RL iter by ``slime/utils/data.py::materialize_lazy_payloads``,
which iterates over every ``Sample.multimodal_lazy_payloads`` entry in
``rollout_data`` and hydrates them in parallel (ThreadPoolExecutor).
The hydrate runs from ``actor._get_rollout_data``, between the ray.get
of rollout_data and any DataIterator setup for log_probs / train.

Each per-image dict carries ``{"image_data": uint8_tensor}`` — PNG-encoded
bytes wrapped in a ``torch.Tensor`` for ray.put cross-Sample dedup. The
tensor wrap is load-bearing: ray serialization de-duplicates ``torch.Tensor``
object identity across multiple references in a single ``ray.put`` graph
but does NOT dedup raw ``bytes``. Without the wrap, the lazy_payloads
design's cross-Sample shared-trajectory image bank gets serialized N times.

``expand`` re-materializes pixel_values on the train side:
  1. Pulls the uint8 tensor out of the per-image dict.
  2. Decodes via ``PIL.Image.open`` (PNG auto-sniffed from magic bytes).
  3. Runs ``processor.image_processor`` to get pixel_values.

Trade-off: avoids shipping precomputed ``pixel_values`` through plasma, at the
cost of a trainer-side materialization pass. Rollout still runs processor logic
to get prompt tokenization and vision-token expansion correct.

The trainer process needs an HF processor to decode + image_processor.
It's lazy-loaded on first call, reading the path from slime's
``args.hf_checkpoint`` (threaded in via the expand-fn call). No env-var
coupling — the processor path tracks ``--hf-checkpoint`` automatically.

Wiring: each ``scripts/train/run_*.sh`` passes
``--multimodal-lazy-expand-fn-path lite.train.utils.multimodal_expand.expand``
to slime. slime's flag defaults to ``None`` (no lite name in slime
itself), and ``materialize_lazy_payloads`` resolves the dotted path
at runtime via ``load_function``.

``expand`` is model-agnostic: it loads whatever processor ``args.hf_checkpoint``
points at (set by ``--hf-checkpoint`` in the run scripts, i.e. by ``MODEL_ID``).
The snippet below just needs *any* local VLM checkpoint to be runnable — point
``HF_CKPT`` at whatever you're training.

Sanity check (needs slime container; substitute your own checkpoint):

    HF_CKPT=/root/models/Qwen/Qwen3-VL-2B-Instruct  # any VLM you're training
    docker exec lite.slime-train-... python3 -c "
    import io, os, torch
    from PIL import Image
    from lite.train.utils.multimodal_expand import expand

    pil = Image.new('RGB', (224, 224), color=(50, 100, 200))
    buf = io.BytesIO(); pil.save(buf, format='PNG')
    payload = {
        'images': {0: {'image_data':
            torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)}},
        'indices': (0,),
    }
    from types import SimpleNamespace
    out = expand(payload, args=SimpleNamespace(hf_checkpoint=os.environ['HF_CKPT']))
    assert 'pixel_values' in out
    print('OK')
    "
"""
from __future__ import annotations

import io
import os
from threading import Lock
from typing import Any

import torch

# ----------------------------------------------------------------------------
# Lazy processor cache.
# ----------------------------------------------------------------------------

#: Trainer-side HF processor, lazy-loaded the first time :func:`expand` runs.
#: The lock guards against concurrent first-touch by multiple micro-batches
#: on the same Megatron rank.
_processor: Any = None
_processor_lock = Lock()


def _get_processor(hf_checkpoint: str | None) -> Any:
    """Return the cached HF processor, loading it on first call.

    Path comes from slime's ``args.hf_checkpoint`` (threaded through
    ``expand(payload, args=args)``). Coupling stays tight by design —
    one variable, no env-var split-brain.
    """
    global _processor
    if _processor is None:
        with _processor_lock:
            if _processor is None:
                assert hf_checkpoint, (
                    "expand() called but slime's ``args.hf_checkpoint`` is None. "
                    "Either the CLI flag wasn't passed, or your slime fork doesn't "
                    "forward args to the expand hook (see "
                    "``slime/backends/megatron_utils/data.py``)."
                )
                from slime.utils.processing_utils import load_processor
                _processor = load_processor(hf_checkpoint, trust_remote_code=True)
                assert _processor is not None, (
                    f"load_processor returned None for {hf_checkpoint!r}; "
                    "see slime.utils.processing_utils.load_processor for hints."
                )
    return _processor


# ENTRY POINT — selected by dotted string, not by a Python import in the
# production path. Training launchers pass
# ``--multimodal-lazy-expand-fn-path lite.train.utils.multimodal_expand.expand``,
# and slime resolves it with ``load_function`` at runtime. Renaming/moving this
# symbol requires updating those launcher strings too.
def expand(payload: dict | None, args: Any = None, **_: Any) -> dict | None:
    """Materialize ``{images, indices}`` into a ready-to-feed dict.

    Slime calls this as ``expand_fn(payload, args=args)`` from
    ``slime/utils/data.py::materialize_lazy_payloads``, which runs once
    per RL iter inside ``actor._get_rollout_data``. ``args`` carries
    ``args.hf_checkpoint`` for the lazy processor load on first use.

    Payload format (built by ``lite/train/rollout/core/segmenter.py::build_segment_samples``):

        images[i]  = {"image_data": uint8_tensor}   # PNG bytes
        indices    = (img_idx, ...)                 # K-window view

    Returns the standard ``multimodal_train_inputs`` shape that
    ``get_batch`` expects:

        {"pixel_values": cat_tensor, "image_grid_thw": cat_tensor, ...}

    ``None`` payload → ``None`` output (defensive; slime only calls this
    when the Sample carries ``multimodal_lazy_payloads``, but we treat
    None robustly so the contract is total).

    Empty indices → empty dict (legitimate: text-only steps may carry
    payload with ``indices=()`` if the rollout protocol decided no
    images were visible).

    ``**_`` swallows any additional kwargs slime might add in the
    future without forcing call-site sync.
    """
    if payload is None:
        return None

    images: dict[int, dict] = payload["images"]
    indices: tuple[int, ...] = payload["indices"]
    if not indices:
        return {}

    missing = [i for i in indices if i not in images]
    if missing:
        raise KeyError(
            f"lazy multimodal expand: indices {indices} reference image "
            f"index {missing[0]} not present in trajectory images dict "
            f"(have keys: {sorted(images.keys())})"
        )

    from PIL import Image

    use_bf16_mm = os.environ.get("CUA_LITE_MULTIMODAL_FP32") != "1"
    processor = _get_processor(getattr(args, "hf_checkpoint", None))

    pils = []
    for i in indices:
        # ``tobytes()`` copies once (tensor → contiguous buffer);
        # ``Image.open`` is lazy until ``load()`` triggers decode.
        encoded = images[i]["image_data"].numpy().tobytes()
        try:
            pil = Image.open(io.BytesIO(encoded))
            pil.load()
        except Exception as e:
            raise RuntimeError(
                f"Failed to PNG-decode lazy multimodal image at index {i} "
                f"(encoded {len(encoded)} bytes, magic={encoded[:4]!r}): {e}"
            ) from e
        pils.append(pil)

    # One batched, HF-native call. Do not dedup ``indices`` here: repeated image
    # indices are a valid ordered processor binding and must be materialized in
    # that exact order. The shared ``processor`` is never mutated.
    out = processor.image_processor(images=pils, return_tensors="pt")

    materialized: dict[str, torch.Tensor] = {}
    for k, v in out.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point() and use_bf16_mm:
            v = v.to(torch.bfloat16)
        materialized[k] = v
    return materialized
