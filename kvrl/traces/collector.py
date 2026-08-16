"""Collect full-cache traces from the real model for the simulator.

For each task instance: run the engine with a full cache, attention statistics on, and an
``on_state`` hook capturing every decision-step's attention rows; greedy-generate ``G``
tokens (also traced); map the task's critical character spans to token indices through
the chat template; save a :class:`Trace`.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from kvrl.cache.view import CacheState
from kvrl.controllers.heuristics import FullCacheController
from kvrl.engine import InferenceEngine
from kvrl.eval.tasks import TaskInstance, is_correct
from kvrl.models.hf_model import HFCausalLM

from .storage import Trace, save_trace


class TracingFullCache(FullCacheController):
    name = "full_traced"
    needs_attention = True


def critical_token_mask(model: HFCausalLM, task: TaskInstance, ids: torch.Tensor) -> np.ndarray:
    """Map ``task.critical_spans`` (chars in task.prompt) to a token mask over ``ids``."""
    tok = model.tokenizer
    T = int(ids.numel())
    mask = np.zeros(T, dtype=bool)
    if not task.critical_spans or tok is None:
        return mask
    msgs = []
    if task.system:
        msgs.append({"role": "system", "content": task.system})
    msgs.append({"role": "user", "content": task.prompt})
    try:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except Exception:
        text = (task.system + "\n\n" if task.system else "") + task.prompt
    start = text.find(task.prompt)
    if start < 0:
        return mask
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    enc_ids = enc["input_ids"]
    if enc_ids[: min(len(enc_ids), T)] != ids[: min(len(enc_ids), T)].tolist():
        # tokenisation mismatch: fall back to prefix-agnostic alignment on the prompt only
        return mask
    offsets = enc["offset_mapping"]
    spans = [(start + a, start + b) for a, b in task.critical_spans]
    for i, (a, b) in enumerate(offsets[:T]):
        if b <= a:
            continue
        for sa, sb in spans:
            if a < sb and b > sa:
                mask[i] = True
                break
    return mask


def answer_token_mask(
    token_ids: np.ndarray, critical: np.ndarray, answers: list[str], tok
) -> np.ndarray:
    """Tokens of the answer string(s) inside the critical span(s) — what actually decides the
    task (a needle's frame vs its code). Used to weight the critical-eviction penalty."""
    from itertools import pairwise

    mask = np.zeros(len(token_ids), dtype=bool)
    if not answers or not critical.any() or tok is None:
        return mask
    idx = np.nonzero(critical)[0]
    spans, start = [], idx[0]
    for a, b in pairwise(idx):
        if b != a + 1:
            spans.append((start, a))
            start = b
    spans.append((start, idx[-1]))
    for s, e in spans:
        pieces = [tok.decode([int(t)]) for t in token_ids[s : e + 1]]
        text = "".join(pieces)
        offs, pos = [], 0
        for piece in pieces:
            offs.append((pos, pos + len(piece)))
            pos += len(piece)
        for ans in answers:
            a = str(ans)
            j = text.find(a)
            while j >= 0:
                for i, (x, y) in enumerate(offs):
                    if x < j + len(a) and y > j:
                        mask[s + i] = True
                j = text.find(a, j + 1)
    return mask


def collect_trace(
    model: HFCausalLM,
    task: TaskInstance,
    trace_id: str,
    *,
    max_new_tokens: int = 128,
    chunk: int = 64,
    out_dir: str | Path | None = None,
    extra_meta: dict | None = None,
) -> Trace:
    ids = model.encode_chat(task.prompt, task.system) if task.system else model.encode(task.prompt)
    if task.task == "lm" and task.continuation:
        # language-modelling traces: force the true continuation instead of generating
        forced = model.encode(task.continuation)[:max_new_tokens]
    else:
        forced = None
    eng = InferenceEngine(model, chunk_size=chunk, decide_every=chunk)
    rows_mean: list[np.ndarray] = []
    rows_max: list[np.ndarray] = []
    step_end: list[int] = []
    step_phase: list[int] = []
    last_state: dict = {}

    def on_state(st: CacheState) -> None:
        rows_mean.append(st.attn_last_mean.numpy().astype(np.float16))
        rows_max.append(st.attn_last_max.numpy().astype(np.float16))
        step_end.append(st.n)
        step_phase.append(st.phase)
        last_state["k"] = st.k_norm
        last_state["v"] = st.v_norm
        last_state["c"] = st.adj_cos

    t0 = time.time()
    res = eng.run(
        ids,
        TracingFullCache(),
        budget=1 << 30,
        max_new_tokens=max_new_tokens,
        forced_ids=forced,
        on_state=on_state,
        force_stats=True,
        stop_on_eos=forced is None,
    )
    elapsed = time.time() - t0
    n_prompt = int(ids.numel())
    gen = res.generated_ids
    # the final generated token is never fed back, so the traced sequence is P + (G-1) long
    # unless forced; store all generated tokens but keep arrays aligned to the cache length.
    T_cache = step_end[-1] if step_end else n_prompt
    all_ids = torch.cat([ids.cpu(), torch.tensor(gen, dtype=torch.long)])[:T_cache]
    K = len(rows_mean)
    attn_mean = np.zeros((K, T_cache), dtype=np.float16)
    attn_lmax = np.zeros((K, T_cache), dtype=np.float16)
    for k, (rm, rx) in enumerate(zip(rows_mean, rows_max)):
        attn_mean[k, : rm.shape[0]] = rm
        attn_lmax[k, : rx.shape[0]] = rx
    knorm = last_state["k"].numpy()[:T_cache]
    vnorm = last_state["v"].numpy()[:T_cache]
    cos = last_state["c"].numpy()[:T_cache]
    crit = critical_token_mask(model, task, ids)
    crit_full = np.zeros(T_cache, dtype=bool)
    crit_full[: min(T_cache, crit.shape[0])] = crit[:T_cache]
    ans_mask = answer_token_mask(all_ids.numpy(), crit_full, task.answers, model.tokenizer)
    correct = is_correct(res.text, task.answers) if task.answers else None
    meta = {
        "task": task.task,
        "answers": task.answers,
        "seed": task.meta.get("seed"),
        "target_tokens": task.meta.get("target_tokens"),
        "task_meta": task.meta,
        "model": model.info.name,
        "dtype": str(model.info.dtype),
        "device": str(model.info.device),
        "chunk": chunk,
        "max_new_tokens": max_new_tokens,
        "forced": forced is not None,
        "generated_text": res.text[:2000],
        "correct_full": correct,
        "elapsed_s": round(elapsed, 2),
        "timings": res.timings,
        **(extra_meta or {}),
    }
    tr = Trace(
        trace_id=trace_id,
        token_ids=all_ids.numpy().astype(np.int32),
        n_prompt=n_prompt,
        n_gen=int(T_cache - n_prompt),
        chunk=chunk,
        attn_mean=attn_mean,
        attn_lmax=attn_lmax,
        step_end=np.array(step_end, dtype=np.int32),
        step_phase=np.array(step_phase, dtype=np.int8),
        key_norm=knorm.astype(np.float16),
        value_norm=vnorm.astype(np.float16),
        adj_key_cos=cos.astype(np.float16),
        gen_logprob=np.array(res.token_logprobs[: T_cache - n_prompt], dtype=np.float16),
        critical_mask=crit_full,
        meta=meta,
        answer_mask=ans_mask,
    )
    if out_dir is not None:
        save_trace(tr, out_dir)
    return tr
