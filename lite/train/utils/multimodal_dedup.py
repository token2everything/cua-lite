"""Model-agnostic image dedup for multimodal step processing.

At history>=2 the per-step prompts of a single trajectory share images across
their overlapping history windows (ordered ``image_indices`` views like
``[0]``, ``[0,1]``, ``[1,2]``, ``[2,3]`` ...), so a naive
``processor(text, images=[...])`` per step re-runs the (expensive) HF
``image_processor`` on every shared image — ~1.9x the unique-image work at
history=2.

This module removes that redundancy WITHOUT touching the model-specific text /
placeholder-expansion logic. It does so with a tiny, fully generic wrapper:

  ``CachingImageProcessor`` wraps a real HF ``image_processor`` and memoizes
  its per-image output by ``id(img)``. When the host processor's ``__call__``
  asks it to process ``images=[a, b, a]`` it computes ``a`` and ``b`` once each,
  then merges the cached per-image ``BatchFeature``\\ s back into one (cat
  tensors on dim 0, extend list fields) — byte-identical to processing the whole
  list at once, because the only per-image-heavy work IS the image_processor and
  it is run once per distinct object.

Usage (see ``apply``): per trajectory, ``copy.copy`` the processor (SHALLOW —
shares tokenizer / config, cheap) and swap ONLY the copy's ``image_processor``
for a ``CachingImageProcessor``. The ORIGINAL processor is never mutated, so the
mechanism is thread-safe across concurrent trajectories and carries ZERO
model-specific code: each step is just ``traj_proc(text, images=[...])`` — the
processor's OWN ``__call__`` runs whatever model-specific placeholder expansion +
tokenization it does, and the caching image_processor transparently dedups the
repeated images underneath it.

To make the per-step fan-out a read-only cache read, ``prime`` the cache
serially over the trajectory's unique images first; the subsequent
``traj_proc(...)`` calls then only READ the cache (no cache writes, no double
compute — the two diagnostic counters do a benign non-atomic increment, but
they are validator-only and feed no training data / control flow).

NO model-specific tokens, no expansion formula, no ``merge_size`` here. Used by
the rollout export path (``lite/train/rollout/core/segmenter.py::_process_step``) — the
overlapping history windows there are the only place repeated images recur (the
trainer-side lazy expand uses the processor's own batched call, not this). The
dim-0 ``cat`` in ``_merge_features`` assumes each image's features are a stack of
fixed-shape units: true for fixed-patch-grid VLMs (Qwen2.5/3-VL, LLaVA,
InternVL), NOT native-resolution processors (Pixtral) or ones whose image work
runs off a non-``__call__`` entry point (Fuyu).

Run: imported as a library (no ``__main__``). Validate by comparing
segmenter outputs with and without ``CUA_LITE_MULTIMODAL_LAZY_EXPAND`` enabled.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature


class CachingImageProcessor:
    """Wrap a real HF ``image_processor``; memoize per-image output by ``id``.

    Drop-in replacement for ``processor.image_processor``: it accepts the exact
    same ``(images=[...], **images_kwargs)`` call the host processor makes,
    returns the exact same merged ``BatchFeature``, and forwards every other
    attribute (e.g. ``merge_size``, ``patch_size``) to the wrapped processor via
    ``__getattr__`` so model-specific code that reads those still works.

    The cache key is ``id(img)`` — within ONE trajectory the per-step image
    lists reference the SAME decoded PIL objects (decoded once in
    ``build_segment_samples``), so repeated history-window references are the
    same Python object and hit the cache. Scope is ONE trajectory (one wrapper
    per ``copy.copy``'d processor); never share a wrapper across trajectories
    (``id`` reuse after GC would alias distinct images).

    Thread-safe for the rollout's ThreadPoolExecutor when ``prime``'d first: the
    cache is filled serially up front and is read-only during the parallel
    phase, and each cached value is a read-only ``BatchFeature`` reused by
    reference.
    """

    def __init__(self, image_processor: Any) -> None:
        self._ip = image_processor
        self._cache: dict[int, BatchFeature] = {}
        # Bookkeeping so callers / validators can assert the dedup actually
        # fired (real computations == unique images, not Σ|images| per step).
        self.real_computations = 0
        self.total_requested = 0

    def prime(self, images: list, **images_kwargs: Any) -> None:
        """Process each (unique-by-id) image up front, serially.

        Call this ONCE before fanning per-step ``processor(...)`` calls out
        across threads: it runs the image_processor exactly once per distinct
        image object and leaves the cache read-only during the parallel phase.

        ``images_kwargs`` MUST match what the host processor later forwards to
        ``__call__`` (e.g. ``return_tensors="pt"``) so the primed per-image
        ``BatchFeature`` is identical to the one a real call would compute.
        Default mirrors ``build_processor_kwargs``' forced ``return_tensors``.
        """
        kwargs = images_kwargs or {"return_tensors": "pt"}
        for img in images:
            key = id(img)
            if key not in self._cache:
                self.real_computations += 1
                self._cache[key] = self._ip(images=[img], **kwargs)

    def __call__(self, images: Any = None, **kwargs: Any) -> Any:
        """Mirror the wrapped image_processor's call, deduping by ``id(img)``.

        ``images`` is whatever the host processor forwards — for the VLMs we run
        that is the step's ``list[PIL.Image]``. Each distinct image is processed
        once; the per-image ``BatchFeature``\\ s are merged in call order. A
        prime'd image is reused as-is: the host processor passes the same
        ``images_kwargs`` for every step of a trajectory, so the cached feature
        matches.
        """
        if images is None:
            return self._ip(images=images, **kwargs)

        img_list = images if isinstance(images, (list, tuple)) else [images]
        self.total_requested += len(img_list)
        feats = []
        for img in img_list:
            key = id(img)
            feat = self._cache.get(key)
            if feat is None:
                self.real_computations += 1
                feat = self._ip(images=[img], **kwargs)
                self._cache[key] = feat
            feats.append(feat)
        return self._merge_features(feats)

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not set on this wrapper (``_ip`` etc. are
        # real instance attrs and short-circuit before this). Forwards
        # model-specific reads like ``merge_size`` to the wrapped processor.
        #
        # Guard the recursion that copy.deepcopy / pickle probing triggers:
        # those probe dunder hooks (``__deepcopy__``, ``__setstate__``, ...) via
        # hasattr BEFORE ``_ip`` is restored on a half-built copy, so a naive
        # ``getattr(self._ip, ...)`` would re-enter ``__getattr__`` for ``_ip``
        # forever. Bail to AttributeError for private/dunder names and when
        # ``_ip`` isn't set yet — the host processor never reads those off the
        # image_processor, and this keeps the wrapper deepcopy-safe.
        if name.startswith("_") or "_ip" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self._ip, name)

    @staticmethod
    def _merge_features(feats: list[BatchFeature]) -> BatchFeature:
        """Concat per-image features in order (cat tensors / extend lists).

        Mirrors how the real image_processor batches multiple images into one
        ``BatchFeature`` (dim-0 cat for ``pixel_values``, row-append for
        ``image_grid_thw`` / list fields). Returns a ``BatchFeature`` so the
        host processor's ``__call__`` sees exactly the type it expects.
        """
        merged: dict[str, Any] = {}
        for key in feats[0].keys():
            vals = [f[key] for f in feats]
            if isinstance(vals[0], torch.Tensor):
                merged[key] = torch.cat(vals, dim=0)
            else:
                out: list = []
                for v in vals:
                    out.extend(v if isinstance(v, (list, tuple)) else [v])
                merged[key] = out
        return BatchFeature(data=merged)


def apply(
    processor: Any,
    unique_images: list | None = None,
    images_kwargs: dict | None = None,
) -> Any:
    """Return a trajectory-scoped processor that dedups repeated images.

    ``copy.copy`` is a SHALLOW copy — it shares the tokenizer / config / image
    token with the original (cheap) but gives us a private ``image_processor``
    slot. We swap ONLY that slot for a :class:`CachingImageProcessor`; the
    ORIGINAL ``processor`` is never mutated, which keeps the original usable
    elsewhere and makes this safe to call concurrently across trajectories.

    Calling ``traj_proc(text, images=[...])`` then runs the processor's OWN
    model-specific ``__call__`` (placeholder expansion + tokenize) with image
    processing transparently deduped underneath — no model-specific code here.

    If ``unique_images`` is given, the cache is primed serially over them so the
    subsequent per-step calls are read-only cache reads (thread-safe fan-out);
    ``images_kwargs`` MUST match what the host processor forwards to the
    image_processor per step (pull it from ``build_processor_kwargs``).
    """
    traj_proc = copy.copy(processor)
    traj_proc.image_processor = CachingImageProcessor(processor.image_processor)
    if unique_images:
        traj_proc.image_processor.prime(unique_images, **(images_kwargs or {}))
    return traj_proc
