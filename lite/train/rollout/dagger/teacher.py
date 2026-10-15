"""DAgger online teacher-forcing — relabel each student step's SFT target.

Mirrors OPD's per-step teacher query (build a teacher adapter from the ``dagger:``
config's ``teacher_agent_id``/``teacher_agent_kwargs``, truncate the trajectory to
turn ``k``'s pre-action context, re-render under the teacher config on the SAME
``lite_sample`` raw images), but with two changes vs the prior OPD reward:

  * the teacher **GENERATES** its own target (``/generate`` sampling) instead of
    **scoring** the student's response (``max_new_tokens=0, return_logprob=True``);
  * the generated tool-call or final-text target **replaces the step's SFT
    target** (``response`` / ``response_tokens``) instead of producing
    per-token reverse-KL logprobs.

So there is no token alignment (we never score the student's tokens) — the teacher
text is re-tokenized in the student's space and EOS-terminated. The student's
``step.prompt`` (its on-policy context: action/tool-result history plus current
user/tool input) is left UNCHANGED; only the *target* becomes the teacher's
output. The teacher output is never executed and never enters the env history —
that stays the student's own.

Wired as the relabel hook of generate() (lite/train/rollout/core/engine.py) via
lite.train.rollout.dagger.generate; skipped at eval.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import re
from typing import Any

from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine

from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.core.adapter import AgentAdapterRegistry, tool_surface_agent_kwarg_names
from lite.core.messages import keep_model_visible_content, peel_system_message
from lite.core.messages.final import MODEL_OUTPUT_ERROR_KEY, no_tool_call_final_text
from lite.core.messages.image_refs import referenced_images_in_message_order
from lite.core.messages.turns import truncate_to_observation
from lite.train.export.sft_tokenize import agent_step_to_rl_step
from lite.utils.registry import rebase_key

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


_DUMMY_USER = {"role": "user", "content": [{"type": "text", "text": ""}]}


def _target_is_empty(tokenizer, response_tokens) -> bool:
    """True when a rendered target carries no trainable content.

    An action that renders to ``''`` does NOT produce an empty token list: the
    chat template still terminates the turn, so the target is the terminator
    alone — measured on the real tokenizers, ``'<|im_end|>\\n'`` (2 tokens) for
    Qwen2.5-VL / Qwen3.5 and ``'<|eot_id|>'`` (1 token) for Llama-3. Both are
    truthy, so a ``not response_tokens`` guard waves them through and the step
    trains on an empty target while reading as relabelled.

    Decoding with ``skip_special_tokens=True`` drops exactly those template
    tokens, leaving ``''`` for an empty target and the action text for a real
    one — which is why the predicate is on the decoded body, not the token count.
    """
    return not tokenizer.decode(response_tokens, skip_special_tokens=True).strip()


def _render_student_step(student_adapter, processor, lite_msg):
    """round_trip: render a parsed teacher action in the student's grammar.

    The shared SFT primitive supplies the student's chat template and turn
    terminator, so callers must not append EOS. ``None`` means the action could
    not be represented as a trainable student target; the caller drops that
    step, not the whole trajectory.
    """
    student_msg = student_adapter.convert_message_to_agent(lite_msg)
    enable_thinking = bool(getattr(student_adapter, "enable_thinking", False))
    rl = agent_step_to_rl_step(
        [_DUMMY_USER, student_msg], processor, enable_thinking=enable_thinking
    )
    if rl is None or _target_is_empty(processor.tokenizer, rl.response_tokens):
        return None
    return rl


def _drop(step) -> bool:
    """Clear a step's target so it's filtered downstream (no SFT sample). Returns False."""
    step.response = ""
    step.response_tokens = []
    step.response_log_probs = []
    return False


def dagger_cfg(args: Any) -> dict:
    """The ``dagger:`` block from the custom-config YAML (``args.dagger``), or ``{}``.

    All DAgger config lives grouped under this one namespace: the teacher agent
    (``teacher_agent_id`` / ``teacher_agent_kwargs``, mirroring the student's top-level
    ``agent_id`` / ``agent_kwargs``) plus the runtime knobs (``teacher_url``,
    ``teacher_temperature``, ``teacher_max_new_tokens``, ``strip_teacher_think``,
    ``relabel_mode``, ``step_filter``).
    """
    return getattr(args, "dagger", None) or {}


# The one value that points the teacher at the served student. Self-distillation
# stays reachable as a null-case control, but it has to be ASKED for by name.
SELF_TEACHER_URL = "self"


def _teacher_url(args: Any) -> str:
    """Teacher sglang ``/generate`` URL.

    Precedence: ``dagger.teacher_url`` in YAML, then ``DAGGER_TEACHER_URL``.
    Either may be the literal ``"self"`` for the explicit self-distill null
    case. There is no implicit fallback to the student server; an absent URL
    usually means a misnested config, missing env var, or failed teacher launch.
    """

    url = dagger_cfg(args).get("teacher_url") or os.environ.get("DAGGER_TEACHER_URL")
    if not url:
        raise ValueError(
            "DAgger has no teacher URL: set `dagger.teacher_url` in the custom-config "
            "YAML (check the keys really are nested under `dagger:`) or export "
            "DAGGER_TEACHER_URL (run_dagger.sh does when NUM_TEACHER_GPUS>0). For the "
            f"self-distill null case ask for it by name: teacher_url='{SELF_TEACHER_URL}'."
        )
    if url == SELF_TEACHER_URL:
        return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    return url


def _truncate_to_observation(lite_sample, k: int):
    """Slice *lite_sample* to turn ``k``'s pre-action context (1-indexed).

    Keep the system message + every message up to and INCLUDING the k-th
    observation block, dropping turn k's assistant and everything after. The
    shared turn helper keeps tool+user observation blocks aligned with the
    agent/protocol turn count.
    """
    system_message, content = peel_system_message(lite_sample.messages)
    kept = truncate_to_observation(content, k)
    final = ([system_message] + kept) if system_message else kept
    return dataclasses.replace(lite_sample, messages=final)


def _build_teacher_adapter(student_adapter, teacher_agent_id: str,
                           teacher_agent_kwargs: dict | None, agent_key: str):
    """Build the teacher adapter from the DAgger config's teacher block. Ported from OPD.

    ``teacher_agent_id`` + ``teacher_agent_kwargs`` mirror the student's ``agent_id`` /
    ``agent_kwargs`` schema. The teacher reuses the student's metadata-derived
    registry suffix (same env) and carries the student adapter's env ``metadata``
    (tools / valid_actions — single source of truth) so the action space matches.
    """
    teacher_key = rebase_key(agent_key, teacher_agent_id)
    kwargs = dict(teacher_agent_kwargs or {})
    surface_overrides = tool_surface_agent_kwarg_names(kwargs)
    if surface_overrides:
        raise ValueError(
            f"dagger.teacher_agent_kwargs contains tool-surface settings "
            f"{sorted(surface_overrides)}; teacher surface comes from the "
            "student/env metadata"
        )
    kwargs["metadata"] = student_adapter.metadata
    return AgentAdapterRegistry.get(teacher_key, **kwargs)


async def relabel_with_teacher(args: Any, rl_sample, sample, agent, agent_key: str):
    """Replace each step's SFT target with the teacher's generated target.

    For every step the student actually produced a response on, query the teacher
    on that step's pre-action context and swap the target. Steps where the
    teacher returns no usable tool-call/final-text target have their target cleared
    (``response_tokens=[]``) so the empty sample is filtered out downstream —
    never fabricate a target. Returns ``rl_sample`` (mutated in place);
    ``step.prompt`` (the student's context) is untouched.
    """
    cfg = dagger_cfg(args)
    teacher_agent_id = cfg.get("teacher_agent_id")
    teacher_agent_kwargs = cfg.get("teacher_agent_kwargs")
    processor = agent.processor
    tokenizer = processor.tokenizer
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(
            "tokenizer has no eos_token_id; DAgger cannot terminate the SFT target"
        )
    url = _teacher_url(args)
    # raw mode keeps teacher <think> by default. For parsed tool-call relabels,
    # round_trip lets the student's own template decide thinking rendering.
    strip_think = bool(cfg.get("strip_teacher_think", False))
    temperature = float(cfg.get("teacher_temperature", 0.0))
    # Default the teacher's decode budget to the student's (same action distribution);
    # override with dagger.teacher_max_new_tokens only to give the teacher more room.
    max_new_tokens = int(
        cfg.get("teacher_max_new_tokens") or args.rollout_max_response_len
    )
    # raw: paste teacher text directly; valid only when teacher and student
    # share response grammar + coordinate space. round_trip: re-render the
    # parsed action in the student's grammar for cross-adapter /
    # cross-resolution teachers.
    relabel_mode = cfg.get("relabel_mode", "raw")
    if relabel_mode not in ("raw", "round_trip"):
        raise ValueError(
            "dagger.relabel_mode must be 'raw' or 'round_trip', "
            f"got {relabel_mode!r}"
        )

    # Cross-config teacher (richer history / resolution), or same-context.
    teacher_adapter = (
        _build_teacher_adapter(agent.adapter, teacher_agent_id, teacher_agent_kwargs, agent_key)
        if teacher_agent_id else None
    )
    # parsing always uses the teacher's (or student's) adapter for action format.
    parse_adapter = teacher_adapter or agent.adapter

    # Per-trajectory image caches (mirrors OPD / engine.py _png_cache): a K-window
    # teacher re-renders the SAME screenshots every step → cache id(raw PIL)→processed
    # and id(img)→PNG-base64 to keep it O(steps) not O(steps²). ids are stable for the
    # lifetime of this call (the trajectory owns the PILs).
    _proc_cache: dict[int, Any] = {}
    _enc_cache: dict[int, str] = {}

    def _render_teacher_prompt(step, turn: int):
        """(prompt_text, raw_images) for the teacher at 0-indexed ``turn``."""
        if teacher_adapter is None:  # same-context fast path
            images = [
                rl_sample.processed_images[i]
                for i in step.image_indices
            ]
            return step.prompt, images
        k = turn + 1  # render_step is 1-indexed
        obs = _truncate_to_observation(rl_sample.lite_sample, k)

        def _proc(im):
            c = _proc_cache.get(id(im))
            if c is None:
                c = teacher_adapter.process_image(im)
                _proc_cache[id(im)] = c
            return c

        teacher_processed = [_proc(image) for image in obs.images]
        agent_step = teacher_adapter.render_step(obs, k, teacher_processed)
        # Model boundary: strip internal side-channel parts (metadata etc.) the
        # protocol preserved — same #42 crash on the strict template otherwise.
        agent_step = keep_model_visible_content(agent_step)
        et = getattr(teacher_adapter, "enable_thinking", False)
        tprompt = processor.apply_chat_template(
            agent_step, add_generation_prompt=True, tokenize=False, enable_thinking=et,
        )
        images = referenced_images_in_message_order(agent_step, teacher_processed)
        return tprompt, images

    async def _relabel_step(step, turn: int) -> bool:
        prompt_text, raw_images = _render_teacher_prompt(step, turn)
        payload: dict = {
            "text": prompt_text,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "skip_special_tokens": False,
            },
        }
        if raw_images:
            async def _enc(im):
                c = _enc_cache.get(id(im))
                if c is None:
                    c = await asyncio.to_thread(encode_image_for_rollout_engine, im)
                    _enc_cache[id(im)] = c
                return c
            payload["image_data"] = list(
                await asyncio.gather(*[_enc(im) for im in raw_images])
            )

        out = await post(url, payload)
        tresp: str = out["text"]
        target = _THINK_RE.sub("", tresp) if strip_think else tresp

        # Gate (both modes): use a parsed tool-call target or a content-only
        # final-text target. Never fabricate a target.
        try:
            agent_msg = parse_adapter.parse_raw_assistant_response(tresp)
            lite_msg = parse_adapter.convert_message_from_agent(agent_msg)
        except ModelToolCallParseError as exc:
            # Malformed TEACHER OUTPUT — the raising twin of the marked-error
            # path below (claude/gpt/fara raise; the qwen families mark), so it
            # drops the step and stays out of the errored count. Any OTHER
            # exception is an adapter/converter defect: it propagates to the
            # gather() handler, which clears the step and counts it errored.
            logger.warning("DAgger relabel: teacher output unparseable, dropped: %r", exc)
            return _drop(step)
        # A parse failure is NOT a content-only final: qwen3_5 / qwen3_vl /
        # qwen2_5_vl mark the error and still leave the raw malformed response as
        # visible text, so ``no_tool_call_final_text`` returns the garbage and it
        # would silently become this step's SFT target.
        if lite_msg.get(MODEL_OUTPUT_ERROR_KEY):
            return _drop(step)
        if not lite_msg.get("tool_calls"):
            final_text = no_tool_call_final_text(lite_msg)
            final_target = _THINK_RE.sub("", final_text) if strip_think else final_text
            if final_target.strip():
                toks = tokenizer.encode(final_target, add_special_tokens=False)
                while toks and toks[-1] == eos_id:
                    toks.pop()
                toks.append(eos_id)
                step.response = final_target
                step.response_tokens = toks
                step.response_log_probs = []
                return True
            return _drop(step)                 # no usable target

        if relabel_mode == "round_trip":
            # Re-render structurally in the student's grammar.
            rl = _render_student_step(agent.adapter, processor, lite_msg)
            if rl is None:  # unrenderable / empty target
                return _drop(step)
            step.response = rl.response
            step.response_tokens = list(rl.response_tokens)
        else:
            # raw: teacher text plus exactly one EOS. Strip any server-kept
            # terminator before appending the tokenizer's EOS.
            toks = tokenizer.encode(target, add_special_tokens=False)
            while toks and toks[-1] == eos_id:
                toks.pop()
            toks.append(eos_id)
            step.response = target
            step.response_tokens = toks
        step.response_log_probs = []  # SFT ignores logprobs
        return True

    scored = [(i, s) for i, s in enumerate(rl_sample.steps) if s.response_tokens]
    # One teacher/request/render failure costs that step, not the trajectory;
    # counters below still separate raised errors from other unusable targets.
    results = await asyncio.gather(
        *[_relabel_step(s, i) for i, s in scored], return_exceptions=True
    )
    n_ok = n_dropped = n_errored = 0
    for (_, step), result in zip(scored, results):
        if isinstance(result, BaseException):
            if not isinstance(result, Exception):  # CancelledError et al: not ours
                raise result
            # Clear the target too: a step left holding the STUDENT's own response
            # would silently train self-imitation instead of the teacher's action.
            _drop(step)
            n_dropped += 1
            n_errored += 1
            logger.warning("DAgger relabel: step failed, dropped: %r", result)
        elif result:
            n_ok += 1
        else:
            n_dropped += 1
    # Stash per-trajectory counts so convert_samples_to_train_data can aggregate
    # the historical ``dagger_unparseable_rate`` metric. ``dagger_n_dropped``
    # includes unusable teacher targets and raised request/render failures;
    # ``dagger_n_errored`` is the raised-error subset.
    sample.metadata = sample.metadata or {}
    sample.metadata["dagger_n_scored"] = len(scored)
    sample.metadata["dagger_n_dropped"] = n_dropped
    sample.metadata["dagger_n_errored"] = n_errored
    logger.info(
        "DAgger relabel: %d/%d steps got a teacher target (%d dropped, of which %d errored)",
        n_ok, len(scored), n_dropped, n_errored,
    )
    return rl_sample
