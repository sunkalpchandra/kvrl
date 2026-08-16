"""Inference engine: real model + real KV cache + a controller deciding what to keep.

    tokens ─▶ chunked prefill ─▶ [controller.decide ─▶ physical eviction] ─▶ decode ─▶ …

One :class:`InferenceEngine` per model. :meth:`run` executes a prompt under a controller
and a token budget and returns a :class:`GenerationResult` with generated tokens, a full
latency breakdown (model / controller / compaction), memory numbers (analytic KV bytes and
device allocator), and per-decision retention records for visualisation.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import torch

from kvrl.cache.view import CacheState, KVCacheView
from kvrl.controllers.base import KVCacheController
from kvrl.models.hf_model import HFCausalLM
from kvrl.utils.device import PeakTracker, memory_stats, synchronize


@dataclass
class Decision:
    step: int
    phase: int
    ctx_len: int
    n_before: int
    n_after: int
    budget: int
    evicted_positions: list[int]
    controller_s: float
    compact_s: float
    importance: list[float] | None = None  # per kept slot, if the controller exposes it


@dataclass
class GenerationResult:
    controller: str
    budget: int
    n_prompt: int
    generated_ids: list[int]
    text: str
    n_decisions: int
    n_evicted_total: int
    final_cache_len: int
    peak_cache_len: int
    kv_bytes_final: int
    kv_bytes_peak: int
    kv_bytes_full: int  # what a full cache would hold at the end
    timings: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    alive: list[bool] = field(default_factory=list)  # per absolute position, at the end
    decisions: list[Decision] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)  # of generated / forced tokens
    stats_enabled: bool = False
    stopped_on_eos: bool = False

    def as_dict(self) -> dict:
        d = asdict(self)
        return d

    @property
    def nll(self) -> float:
        return -sum(self.token_logprobs) / max(1, len(self.token_logprobs))


class InferenceEngine:
    def __init__(
        self,
        model: HFCausalLM,
        chunk_size: int = 64,
        decide_every: int = 64,
        n_sink: int = 4,
        check_finite: bool = False,
    ):
        self.model = model
        self.chunk_size = chunk_size
        self.decide_every = decide_every
        self.n_sink = n_sink
        self.check_finite = check_finite
        self.device = model.device

    def _sync_time(self) -> float:
        synchronize(self.device)
        return time.perf_counter()

    @torch.no_grad()
    def run(
        self,
        prompt_ids: torch.Tensor,
        controller: KVCacheController,
        budget: int,
        max_new_tokens: int = 64,
        *,
        forced_ids: torch.Tensor | None = None,
        record_importance: bool = False,
        episode: int = 0,
        stop_on_eos: bool = True,
        on_state: Callable[[CacheState], None] | None = None,
        force_stats: bool = False,
    ) -> GenerationResult:
        """Run one prompt.

        budget: maximum cache size (slots) after each decision. ``budget >= n_prompt +
        max_new_tokens`` means the cache is never touched.
        forced_ids: teacher-forced continuation (scored, not sampled) — processed in chunks;
        gives the NLL of a fixed continuation under this controller/budget.
        on_state: called with every decision-step CacheState (trace collection / analysis).
        force_stats: capture attention statistics even if the controller does not need them.
        """
        model = self.model
        prompt_ids = prompt_ids.flatten().to(self.device)
        n_prompt = int(prompt_ids.numel())
        stats_on = bool(getattr(controller, "needs_attention", False)) or force_stats
        model.set_stats(stats_on)
        model.stats.reset()
        view = KVCacheView(model, n_sink=self.n_sink)
        controller.reset(
            episode=episode, n_prompt=n_prompt, budget=budget, max_new_tokens=max_new_tokens
        )
        alive: list[bool] = []
        decisions: list[Decision] = []
        timings = {"prefill_s": 0.0, "decode_s": 0.0, "controller_s": 0.0, "compact_s": 0.0}
        peak_len = 0
        n_generated = 0
        generated: list[int] = []
        logprobs: list[float] = []
        step = 0
        eos = model.eos_token_ids

        def decide(n_new: int, phase: int, ctx_len: int) -> None:
            nonlocal step, peak_len
            n_before = view.n
            peak_len = max(peak_len, n_before)
            state = view.state(
                n_new=n_new,
                step=step,
                ctx_len=ctx_len,
                phase=phase,
                n_prompt=n_prompt,
                n_generated=n_generated,
                max_new_tokens=max_new_tokens,
                accumulate=True,
            )
            if on_state is not None:
                on_state(state)
            if n_before <= budget:
                # nothing to evict; controllers with per-slot memory still see the state
                controller.on_compact(torch.arange(n_before), n_before)
                step += 1
                return
            t0 = self._sync_time()
            keep = controller.decide(state, budget).detach().cpu()
            t1 = self._sync_time()
            imp = None
            if record_importance and hasattr(controller, "importance"):
                imp = controller.importance(state).float().cpu().index_select(0, keep).tolist()
            evicted_mask = torch.ones(n_before, dtype=torch.bool)
            evicted_mask[keep] = False
            evicted_pos = view.positions[evicted_mask].tolist()
            view.compact(keep)
            controller.on_compact(keep, n_before)
            t2 = self._sync_time()
            for p in evicted_pos:
                alive[p] = False
            timings["controller_s"] += t1 - t0
            timings["compact_s"] += t2 - t1
            decisions.append(
                Decision(
                    step,
                    phase,
                    ctx_len,
                    n_before,
                    view.n,
                    budget,
                    evicted_pos,
                    t1 - t0,
                    t2 - t1,
                    imp,
                )
            )
            step += 1

        with PeakTracker(self.device) as peak_mem:
            # ---------------------------------------------------------------- prefill
            t_start = self._sync_time()
            pos = 0
            logits = None
            for s in range(0, n_prompt, self.chunk_size):
                chunk = prompt_ids[s : s + self.chunk_size]
                q = int(chunk.numel())
                positions = torch.arange(pos, pos + q, device=self.device)
                logits = model.forward_chunk(chunk, positions, view.cache, logits_to_keep=1)
                view.append(positions, is_generated=False, step=step)
                alive.extend([True] * q)
                pos += q
                if self.check_finite and not torch.isfinite(logits).all():
                    raise FloatingPointError("non-finite logits during prefill")
                timings["prefill_s"] += self._sync_time() - t_start
                decide(n_new=q, phase=0, ctx_len=pos)
                t_start = self._sync_time()
                peak_mem.sample()
            timings["prefill_s"] += self._sync_time() - t_start
            assert logits is not None
            # ---------------------------------------------------------------- decode
            t_start = self._sync_time()
            stopped = False
            if forced_ids is not None:
                forced = forced_ids.flatten().to(self.device)
                # score forced tokens in chunks: logits for token t come from the previous step
                prev_logits = logits[:, -1, :]
                for s in range(0, int(forced.numel()), self.decide_every):
                    chunk = forced[s : s + self.decide_every]
                    q = int(chunk.numel())
                    # log-prob of chunk[0] from prev_logits, of chunk[i] from this forward's i-1
                    lp0 = torch.log_softmax(prev_logits.float(), dim=-1)[0, chunk[0]].item()
                    positions = torch.arange(pos, pos + q, device=self.device)
                    out = model.forward_chunk(chunk, positions, view.cache, logits_to_keep=q)
                    view.append(positions, is_generated=True, step=step)
                    alive.extend([True] * q)
                    pos += q
                    lps = torch.log_softmax(out[0, :-1, :].float(), dim=-1)
                    tgt = chunk[1:]
                    got = lps.gather(1, tgt[:, None]).flatten().tolist() if q > 1 else []
                    logprobs.extend([lp0, *got])
                    generated.extend(chunk.tolist())
                    n_generated += q
                    prev_logits = out[:, -1, :]
                    timings["decode_s"] += self._sync_time() - t_start
                    decide(n_new=q, phase=1, ctx_len=pos)
                    t_start = self._sync_time()
                    peak_mem.sample()
            else:
                next_tok = int(logits[0, -1].argmax().item())
                lp = torch.log_softmax(logits[0, -1].float(), dim=-1)[next_tok].item()
                since_decision = 0
                for t in range(max_new_tokens):
                    generated.append(next_tok)
                    logprobs.append(lp)
                    n_generated += 1
                    if stop_on_eos and next_tok in eos:
                        stopped = True
                        break
                    if t == max_new_tokens - 1:
                        break  # last token produced; no need to feed it back
                    positions = torch.tensor([pos], device=self.device)
                    tok = torch.tensor([next_tok], device=self.device)
                    logits = model.forward_chunk(tok, positions, view.cache, logits_to_keep=1)
                    view.append(positions, is_generated=True, step=step)
                    alive.append(True)
                    pos += 1
                    since_decision += 1
                    if since_decision == self.decide_every:
                        timings["decode_s"] += self._sync_time() - t_start
                        decide(n_new=since_decision, phase=1, ctx_len=pos)
                        since_decision = 0
                        t_start = self._sync_time()
                        peak_mem.sample()
                    next_tok = int(logits[0, -1].argmax().item())
                    lp = torch.log_softmax(logits[0, -1].float(), dim=-1)[next_tok].item()
                if since_decision > 0:
                    # final flush: lets tracing/analysis see the last decode tokens' attention
                    timings["decode_s"] += self._sync_time() - t_start
                    decide(n_new=since_decision, phase=1, ctx_len=pos)
                    t_start = self._sync_time()
            timings["decode_s"] += self._sync_time() - t_start
            peak_len = max(peak_len, view.n)
        timings["total_s"] = sum(v for k, v in timings.items() if k != "total_s")
        timings["model_s"] = timings["prefill_s"] + timings["decode_s"]
        n_out = len(generated)
        timings["decode_tok_per_s"] = (
            n_out / timings["decode_s"] if timings["decode_s"] > 0 else 0.0
        )
        timings["prefill_tok_per_s"] = (
            n_prompt / timings["prefill_s"] if timings["prefill_s"] > 0 else 0.0
        )
        info = model.info
        mem = memory_stats(self.device).as_dict()
        mem["peak_allocated_mb"] = peak_mem.peak_bytes / 2**20
        text = model.decode(generated) if model.tokenizer is not None else ""
        model.set_stats(False)
        return GenerationResult(
            controller=controller.name,
            budget=budget,
            n_prompt=n_prompt,
            generated_ids=generated,
            text=text,
            n_decisions=len(decisions),
            n_evicted_total=view.total_evicted,
            final_cache_len=view.n,
            peak_cache_len=peak_len,
            kv_bytes_final=info.kv_bytes(view.n),
            kv_bytes_peak=info.kv_bytes(peak_len),
            kv_bytes_full=info.kv_bytes(pos),
            timings=timings,
            memory=mem,
            alive=alive,
            decisions=decisions,
            token_logprobs=logprobs,
            stats_enabled=stats_on,
            stopped_on_eos=stopped,
        )


def budget_from_fraction(frac: float, n_prompt: int, chunk: int = 64, min_tokens: int = 128) -> int:
    """Token budget = frac × prompt length, rounded to a chunk multiple, at least min_tokens."""
    if frac >= 1.0:
        return 1 << 30
    b = int(math.ceil(frac * n_prompt / chunk) * chunk)
    return max(min_tokens, b)
