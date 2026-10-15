"""Trajectory-to-``slime.Sample`` segmentation for RL/SFT rollouts.

This env-free path turns a multi-turn ``LiteRLSample`` into one or more training
samples. Consecutive turns are packed only when text tokens and image-slot
indices are both prefix-extensional; otherwise they stay as length-1 samples.
"""

from __future__ import annotations

import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lite.core.samples import LiteRLSample  # noqa: F401  # forward-ref annotations only

import torch
from slime.utils.processing_utils import build_processor_kwargs
from slime.utils.types import Sample

from lite.core.samples import (
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_TRUNCATED,
    STEP_STATUSES_BY_SEVERITY,
)
from lite.train.utils import multimodal_dedup

logger = logging.getLogger(__name__)


# Mapping from cua-lite status strings → slime's Sample.Status enum.
# Kept here so cua-lite types (LiteRLStep) stay slime-import-free.
_STATUS_STR_TO_SLIME = {
    STATUS_COMPLETED: Sample.Status.COMPLETED,
    STATUS_TRUNCATED: Sample.Status.TRUNCATED,
    STATUS_ABORTED: Sample.Status.ABORTED,
    STATUS_FAILED: Sample.Status.FAILED,
}

# Severity rank per status, derived from the owner vocabulary. A status absent
# from it is out of vocabulary and must raise rather than rank below
# ``completed``.
_STATUS_SEVERITY = {
    status: rank for rank, status in enumerate(STEP_STATUSES_BY_SEVERITY)
}


# ---------------------------------------------------------------------------
# Lazy multimodal mode announce flag (module-level once-only).
# Set the first time ``build_segment_samples`` runs in a process so the
# rollout log carries one obvious line about which path is active.
# ---------------------------------------------------------------------------

_lazy_mode_logged = False


def _log_lazy_mode_once(lazy_expand: bool, processor) -> None:
    global _lazy_mode_logged
    if _lazy_mode_logged:
        return
    _lazy_mode_logged = True
    if not processor:
        logger.info("Multimodal data flow: text-only path (no images).")
    elif lazy_expand:
        logger.info(
            "Multimodal data flow: LAZY EXPAND — shared trajectory_images dict "
            "carries PNG-encoded image bytes; trainer expand_fn decodes once per "
            "RL iter in actor._get_rollout_data (materialize_lazy_payloads)."
        )
    else:
        logger.info(
            "Multimodal data flow: EAGER (pre-computed pixel_values per Sample, "
            "no shared dict)."
        )


def _emit_rl_sample(
    segment,                              # list[LiteRLStep] — length ≥ 1
    *,
    original_sample: Sample,
    rl_sample: LiteRLSample,            # for trajectory-level fields
    per_step_prompt_ids,                  # list[list[int]] aligned with segment
    multimodal_train_inputs: dict | None,
    multimodal_lazy_payloads: dict | None = None,
    turn_range: tuple[int, int] | None = None,
) -> Sample:
    """Build one training sample from a prefix-extensional turn segment.

    The last prompt already contains prior packed turns, so the emitted token
    sequence is ``last.prompt + last.response``. ``loss_mask`` marks only each
    assistant response span; prompt/context deltas stay masked out. Length-1
    segments keep the original per-turn sample shape.
    """
    initial_prompt_len = len(per_step_prompt_ids[0])
    last_prompt_ids = list(per_step_prompt_ids[-1])
    last_response_tokens = list(segment[-1].response_tokens)
    tokens = last_prompt_ids + last_response_tokens
    response_length = len(tokens) - initial_prompt_len

    # Multi-span loss_mask: indexed from position ``initial_prompt_len``
    # (= 0 in mask coords). Mark each step i's response positions as 1;
    # leave inter-step prompt/context deltas as 0.
    loss_mask = [0] * response_length
    rollout_log_probs = [0.0] * response_length
    for i, step in enumerate(segment):
        # step i's response sits at absolute positions
        # [step_i.prompt_len, step_i.prompt_len + step_i.response_len).
        start = len(per_step_prompt_ids[i]) - initial_prompt_len
        end = start + len(step.response_tokens)
        for j in range(start, end):
            loss_mask[j] = 1
        for off, lp in enumerate(step.response_log_probs):
            rollout_log_probs[start + off] = lp

    response_text_parts = [s.response for s in segment]
    step_statuses = [s.status for s in segment]

    n_turns = len(rl_sample.steps)
    if turn_range is None:
        first = next(
            (i for i, step in enumerate(rl_sample.steps) if step is segment[0]),
            None,
        )
        if first is None:
            raise ValueError("segment[0] is not a step object from rl_sample.steps")
        last = first + len(segment) - 1
    else:
        first, last = turn_range

    # Status: max-severity over the segment. The rank lookup subscripts
    # ``_STATUS_SEVERITY`` directly, so an out-of-vocabulary status raises
    # KeyError here instead of being outranked by a legal sibling status and
    # training a corrupted turn as a complete one.
    worst = max(step_statuses, key=lambda s: _STATUS_SEVERITY[s])
    status = _STATUS_STR_TO_SLIME[worst]

    sample_metadata = dict(original_sample.metadata or {})
    sample_metadata["others"] = {
        **(sample_metadata.get("others") or {}),
        "episode_return": rl_sample.episode_return,
    }
    sample_metadata.update({
        "n_turns": n_turns,
        "turn_range": (first, last),
        "step_statuses": tuple(step_statuses),
    })

    return Sample(
        group_index=original_sample.group_index,
        index=original_sample.index,
        # Per-trajectory aggregation id: every segment of this trajectory shares
        # ``original_sample.index`` (globally unique per trajectory), so slime's
        # depth-3 validator passes and the loss reducer normalizes per trajectory
        # via group_mask_sums.
        group_id=original_sample.index,
        prompt=original_sample.prompt,
        tokens=tokens,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        response_length=response_length,
        response="".join(response_text_parts),
        label=original_sample.label,
        reward=float(segment[-1].reward),
        status=status,
        multimodal_train_inputs=multimodal_train_inputs,
        multimodal_lazy_payloads=multimodal_lazy_payloads,
        metadata=sample_metadata,
    )


# ---------------------------------------------------------------------------
# Build training samples from agent.sample() result
# ---------------------------------------------------------------------------

def _segment_steps(
    per_step_tokens: list[dict],
) -> list[list[int]]:
    """Greedily pack adjacent turns that are text- and image-prefix compatible.

    Text prefix alone is insufficient for VLMs: two same-sized images can render
    to identical image-token ids while their pixels differ. The ordered
    ``image_indices`` prefix check preserves processor image-slot binding.
    """
    if not per_step_tokens:
        return []
    segments: list[list[int]] = [[0]]
    for i in range(1, len(per_step_tokens)):
        prev = per_step_tokens[i - 1]
        curr = per_step_tokens[i]
        # Token-level prefix.
        expected = prev["prompt_tokens"] + prev["response_tokens"]
        L = len(expected)
        token_prefix_ok = curr["prompt_tokens"][:L] == expected
        # Image-index-level prefix.
        prev_imgs = tuple(prev["step"].image_indices or ())
        curr_imgs = tuple(curr["step"].image_indices or ())
        image_prefix_ok = curr_imgs[:len(prev_imgs)] == prev_imgs
        if token_prefix_ok and image_prefix_ok:
            segments[-1].append(i)
        else:
            segments.append([i])
    return segments


def build_segment_samples(
    rl_sample: LiteRLSample,
    original_sample: Sample,
    processor,
    tokenizer,
    state,
) -> list[Sample]:
    """Convert a non-empty rollout trajectory into packed training samples."""

    # ``image_indices`` is an ordered binding to processor image slots. This
    # function consumes it verbatim; producers own validation and ordering.
    #
    # Floating multimodal tensors are cast to bf16 by default to reduce plasma
    # and rollout-to-trainer transfer size. Integer tensors stay unchanged.
    # Set CUA_LITE_MULTIMODAL_FP32=1 to debug dtype regressions.
    use_bf16_mm = os.environ.get("CUA_LITE_MULTIMODAL_FP32") != "1"

    # Lazy multimodal expand stores shared PNG bytes per trajectory and lets the
    # trainer rebuild pixel tensors. It saves plasma at the cost of train-time
    # image processing, so the eager pixel_values path remains the default.
    lazy_expand = os.environ.get("CUA_LITE_MULTIMODAL_LAZY_EXPAND") == "1"
    _log_lazy_mode_once(lazy_expand, processor)

    # The slime expand hook is required when Samples carry
    # ``multimodal_lazy_payloads``. Without it the payloads silently flow
    # through plasma and fail during trainer-side materialization.
    if lazy_expand and processor:
        args = getattr(state, "args", None)
        if args is None or not getattr(args, "multimodal_lazy_expand_fn_path", None):
            raise RuntimeError(
                "multimodal_lazy_payloads requires --multimodal-lazy-expand-fn-path "
                "to be set on the slime CLI (point it at "
                "lite.train.utils.multimodal_expand.expand). "
                "Either pass the flag or set CUA_LITE_MULTIMODAL_LAZY_EXPAND=0 to skip "
                "the lazy path entirely."
            )

    # Lazy path: every emitted Sample shares this trajectory-scoped image dict.
    # The uint8 tensor wrapper is what Ray/cloudpickle can deduplicate.
    trajectory_images: dict[int, dict] | None = None
    if lazy_expand and processor:
        trajectory_images = {}
        referenced_indices = sorted(
            {i for step in rl_sample.steps for i in step.image_indices}
        )
        for img_idx in referenced_indices:
            img = rl_sample.processed_images[img_idx]
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            # ``bytearray`` then ``frombuffer`` gives us a writable tensor
            # that owns its buffer; using ``bytes`` directly would produce
            # a read-only view that pytorch warns about and that ray would
            # copy on serialize anyway.
            trajectory_images[img_idx] = {
                "image_data": torch.frombuffer(
                    bytearray(buf.getvalue()), dtype=torch.uint8,
                ),
            }

    # Per-step processing reuses a trajectory-local processor copy whose image
    # processor cache is primed once per unique image. Step calls are then
    # read-only and order-preserving under ThreadPoolExecutor.map.
    traj_proc = None
    if processor:
        # ``images_kwargs`` (e.g. forced ``return_tensors="pt"``) is what the
        # processor forwards to the image_processor — prime the cache with it so
        # primed per-image features match the per-step calls below.
        images_kwargs = build_processor_kwargs({"images": None})["images_kwargs"]
        unique_imgs = list({
            i: rl_sample.processed_images[i]
            for step in rl_sample.steps for i in step.image_indices
        }.values())
        traj_proc = multimodal_dedup.apply(
            processor, unique_images=unique_imgs, images_kwargs=images_kwargs,
        )

    def _process_step(step) -> dict:
        if step.image_indices:
            if traj_proc is None:
                raise RuntimeError(
                    "LiteRLStep has image_indices but no multimodal processor; "
                    "image steps must use the multimodal branch."
            )
            step_images = [
                rl_sample.processed_images[i]
                for i in step.image_indices
            ]
            # build_processor_kwargs owns the model-specific image argument
            # shape; the cached processor deduplicates repeated images.
            out = traj_proc(
                text=step.prompt, **build_processor_kwargs({"images": step_images}),
            )
            ids = out["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            prompt_ids = list(ids)
            image_inputs = {
                k: v for k, v in out.items()
                if k not in ("input_ids", "attention_mask")
            }

            if lazy_expand:
                # Lazy path keeps only shared image bytes plus this step's view.
                multimodal_train_inputs = None
                multimodal_lazy_payloads = {
                    "images": trajectory_images,
                    "indices": tuple(step.image_indices),
                }
            else:
                # Eager path: per-Sample concatenated pixel_values.
                multimodal_train_inputs = {}
                for k, v in image_inputs.items():
                    if (
                        use_bf16_mm
                        and isinstance(v, torch.Tensor)
                        and v.is_floating_point()
                    ):
                        v = v.to(torch.bfloat16)
                    multimodal_train_inputs[k] = v
                multimodal_train_inputs = multimodal_train_inputs or None
                multimodal_lazy_payloads = None
        else:
            # Text-only step: reuse prompt_tokens stored at SFT export when
            # present, else encode (online rollout stores None).
            stored = step.prompt_tokens
            if stored is not None:
                prompt_ids = stored
            else:
                prompt_ids = tokenizer.encode(step.prompt, add_special_tokens=False)
            multimodal_train_inputs = None
            multimodal_lazy_payloads = None
        return {
            "prompt_tokens": list(prompt_ids),
            "response_tokens": list(step.response_tokens),
            "multimodal_train_inputs": multimodal_train_inputs,
            "multimodal_lazy_payloads": multimodal_lazy_payloads,
            "step": step,
        }

    per_step_tokens: list[dict]
    _proc_workers = int(os.environ.get("CUA_LITE_ROLLOUT_PROC_WORKERS", "8"))
    if processor and _proc_workers > 1 and len(rl_sample.steps) > 1:
        with ThreadPoolExecutor(
            max_workers=min(_proc_workers, len(rl_sample.steps))
        ) as _ex:
            per_step_tokens = list(_ex.map(_process_step, rl_sample.steps))
    else:
        per_step_tokens = [_process_step(step) for step in rl_sample.steps]

    # CUA_LITE_DISABLE_RADIX=1 emits one Sample per turn. GSPO sets this because
    # its ratio is per sequence; GRPO stays packed because its objective is
    # per-token.
    if os.environ.get("CUA_LITE_DISABLE_RADIX") == "1":
        segments_idx = [[i] for i in range(len(per_step_tokens))]
    else:
        segments_idx = _segment_steps(per_step_tokens)

    # (3) Emit one Sample per segment.
    samples: list[Sample] = []
    for seg_indices in segments_idx:
        seg_steps = [per_step_tokens[i]["step"] for i in seg_indices]
        seg_prompt_ids = [per_step_tokens[i]["prompt_tokens"] for i in seg_indices]
        # The last step's multimodal view covers the whole packed segment by the
        # prefix invariant checked in _segment_steps.
        multimodal_train_inputs = per_step_tokens[seg_indices[-1]]["multimodal_train_inputs"]
        multimodal_lazy_payloads = per_step_tokens[seg_indices[-1]]["multimodal_lazy_payloads"]
        samples.append(_emit_rl_sample(
            segment=seg_steps,
            original_sample=original_sample,
            rl_sample=rl_sample,
            per_step_prompt_ids=seg_prompt_ids,
            multimodal_train_inputs=multimodal_train_inputs,
            multimodal_lazy_payloads=multimodal_lazy_payloads,
            turn_range=(seg_indices[0], seg_indices[-1]),
        ))

    # Cooperative abort fixup.
    if state.aborted and samples:
        samples[-1].status = Sample.Status.ABORTED

    return samples
