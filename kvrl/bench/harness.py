"""Benchmark harness: latency and memory vs context length, controller and budget.

Every measurement is a full engine run (prefill + decode) with device synchronisation;
we repeat R times after W warmup runs and report medians + IQR. Reported separately:
model time (prefill/decode), controller time, cache-compaction time, and their sum, plus
analytic KV bytes (peak / final) and the device allocator's peak.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pandas as pd
import torch

from kvrl.controllers import make_controller
from kvrl.engine import InferenceEngine, budget_from_fraction
from kvrl.eval.corpus import Filler, load_corpus
from kvrl.models.hf_model import HFCausalLM
from kvrl.utils.device import empty_cache, synchronize
from kvrl.utils.timing import Samples


def make_prompt(model: HFCausalLM, n_tokens: int, seed: int = 0) -> torch.Tensor:
    """Natural-text prompt of exactly n_tokens tokens (attention patterns matter for timing)."""
    filler = Filler(load_corpus(), seed=seed)
    text = filler.sample(int(n_tokens * 5))
    ids = model.encode(text)
    while ids.numel() < n_tokens:
        text += " " + filler.sample(int((n_tokens - ids.numel()) * 5) + 200)
        ids = model.encode(text)
    return ids[:n_tokens]


def bench(
    model: HFCausalLM,
    cfg: dict,
    log: Callable[[str], None] = print,
    on_row: Callable[[dict], None] | None = None,
) -> tuple[pd.DataFrame, dict]:
    bcfg = cfg["benchmark"]
    lengths = [int(x) for x in bcfg["context_lengths"]]
    warmup, repeats = int(bcfg.get("warmup", 1)), int(bcfg.get("repeats", 3))
    max_new = int(bcfg.get("max_new_tokens", 32))
    engine = InferenceEngine(
        model, chunk_size=int(bcfg.get("chunk", 64)), decide_every=int(bcfg.get("decide_every", 64))
    )
    ctrl_specs = bcfg["controllers"]
    rows = []
    for L in lengths:
        if L > model.info.max_context:
            log(f"[bench] skip {L}: exceeds model max context {model.info.max_context}")
            continue
        try:
            ids = make_prompt(model, L, seed=int(bcfg.get("seed", 0)))
        except Exception as e:  # pragma: no cover
            log(f"[bench] skip {L}: prompt build failed {e!r}")
            continue
        for spec in ctrl_specs:
            spec = dict(spec)
            name = spec.pop("name")
            label = spec.pop("label", name)
            budgets = [1.0] if name == "full" else [float(b) for b in bcfg["budget_fracs"]]
            ctrl = make_controller(name, **spec)
            for bf in budgets:
                budget = budget_from_fraction(bf, L, chunk=engine.chunk_size)
                samples: dict[str, Samples] = {}
                ok = True
                for r in range(warmup + repeats):
                    try:
                        empty_cache(model.device)
                        res = engine.run(
                            ids, ctrl, budget=budget, max_new_tokens=max_new, stop_on_eos=False
                        )
                    except (RuntimeError, MemoryError) as e:
                        log(
                            f"[bench] {label} L={L} b={bf}: FAILED {type(e).__name__}: "
                            f"{str(e)[:80]}"
                        )
                        ok = False
                        break
                    if r < warmup:
                        continue
                    t = res.timings
                    n_out = max(1, len(res.generated_ids))
                    vals = {
                        "prefill_s": t["prefill_s"],
                        "decode_ms_per_tok": 1000 * t["decode_s"] / n_out,
                        "controller_s": t["controller_s"],
                        "compact_s": t["compact_s"],
                        "total_s": t["total_s"],
                        "model_s": t["model_s"],
                        "prefill_tok_per_s": t["prefill_tok_per_s"],
                        "peak_allocated_mb": res.memory.get("peak_allocated_mb") or 0.0,
                    }
                    for k, v in vals.items():
                        samples.setdefault(k, Samples()).add(float(v))
                    kv_peak, kv_full, peak_len = (
                        res.kv_bytes_peak,
                        res.kv_bytes_full,
                        res.peak_cache_len,
                    )
                    stats_on = res.stats_enabled
                if not ok or not samples:
                    rows.append({"context": L, "controller": label, "budget_frac": bf, "ok": False})
                    continue
                row = {
                    "context": L,
                    "controller": label,
                    "budget_frac": bf,
                    "budget": budget,
                    "ok": True,
                    "kv_bytes_peak": kv_peak,
                    "kv_bytes_full": kv_full,
                    "kv_peak_frac": kv_peak / max(1, kv_full),
                    "peak_cache_len": peak_len,
                    "stats_enabled": stats_on,
                    "repeats": repeats,
                }
                for k, s in samples.items():
                    summ = s.summary()
                    row[f"{k}_median"] = summ["median"]
                    row[f"{k}_p25"] = summ["p25"]
                    row[f"{k}_p75"] = summ["p75"]
                rows.append(row)
                if on_row:
                    on_row(row)
                log(
                    f"[bench] L={L:6d} {label:8s} b={bf:<5} prefill {row['prefill_s_median']:.2f}s "
                    f"decode {row['decode_ms_per_tok_median']:.1f} ms/tok "
                    f"ctrl {row['controller_s_median']:.3f}s "
                    f"compact {row['compact_s_median']:.3f}s kv%={row['kv_peak_frac']:.2f} "
                    f"peak {row['peak_allocated_mb_median']:.0f}MB"
                )
    df = pd.DataFrame(rows)
    summary = {
        "rows": len(df),
        "ok": int(df["ok"].sum()) if len(df) else 0,
        "lengths": lengths,
        "device": str(model.device),
        "dtype": str(model.info.dtype),
    }
    return df, summary


def decode_vs_cache_length(
    model: HFCausalLM, lengths: list[int], n_decode: int = 32, repeats: int = 3, log=print
) -> pd.DataFrame:
    """Pure hardware curve: decode ms/token as a function of cache length (full cache)."""
    rows = []
    for L in lengths:
        ids = make_prompt(model, L)
        cache = model.new_cache()
        # chunked prefill (a one-shot 16K forward materialises a ~7 GB score matrix on MPS
        # and swaps; that would contaminate the decode measurement) + pool release
        for s in range(0, L, 256):
            ch = ids[s : s + 256]
            model.forward_chunk(ch, torch.arange(s, s + ch.numel()), cache)
            if model.device.type == "mps" and (s // 256) % 8 == 7:
                empty_cache(model.device)
        empty_cache(model.device)
        tok = ids[-1:]
        s = Samples()
        for r in range(repeats + 1):
            import copy

            c = copy.deepcopy(cache)
            synchronize(model.device)
            t0 = time.perf_counter()
            for i in range(n_decode):
                model.forward_chunk(tok, torch.tensor([L + i]), c)
            synchronize(model.device)
            if r > 0:
                s.add(1000 * (time.perf_counter() - t0) / n_decode)
            del c
        summ = s.summary()
        rows.append(
            {
                "cache_len": L,
                "decode_ms_per_tok_median": summ["median"],
                "p25": summ["p25"],
                "p75": summ["p75"],
                "kv_bytes": model.info.kv_bytes(L),
            }
        )
        log(
            f"[bench] cache {L:6d}: {summ['median']:.1f} ms/tok "
            f"(IQR {summ['p25']:.1f}-{summ['p75']:.1f})"
        )
        del cache
        empty_cache(model.device)
    return pd.DataFrame(rows)


__all__ = ["bench", "decode_vs_cache_length", "make_prompt"]
