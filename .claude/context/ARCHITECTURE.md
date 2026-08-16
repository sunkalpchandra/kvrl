# ARCHITECTURE (authoritative — reconciled 2026-08-16 from design_ml_architect.md + design_inference_engineer.md)

## 1. System overview

```
                 ┌──────────────────────────────────────────────────────────┐
                 │  kvrl.models  (HF causal LM, custom "kvrl" attention)     │
   tokens ──────▶│  chunked prefill (C tokens/step) → DynamicCache          │
                 │  per-layer attention-mass stats (bounded memory)         │
                 └───────────────┬───────────────────────┬──────────────────┘
                                 │ stats buffer          │ K/V tensors
                                 ▼                       ▼
                 ┌────────────────────────┐   ┌────────────────────────────┐
                 │ kvrl.features          │   │ kvrl.cache                 │
                 │ FeatureState (shared   │   │ KVCacheView (slot→pos,     │
                 │ by sim + real)         │   │ per-slot meta), compact(), │
                 └───────────┬────────────┘   │ MaskedReference (test)     │
                             │ obs             └──────────────▲─────────────┘
                             ▼                                │ keep slots
                 ┌───────────────────────────────────────────┴──────────────┐
                 │ kvrl.controllers  KVCacheController.decide(state, budget) │
                 │ full | window | random | snapkv | h2o | tova | keynorm |  │
                 │ hybrid | regressor | oracle(sim) | rl (PPO policy)        │
                 └──────────────────────────────────────────────────────────┘

   Env A (kvrl.sim):  recorded traces ──▶ FeatureState ──▶ policy ──▶ reward from trace
   Env B (kvrl.models + kvrl.cache): live model ──▶ FeatureState ──▶ policy ──▶ physical eviction
```

## 2. Model stack (kvrl/models)

- Primary model `qwen2.5-0.5b-instruct` (24 layers, 14 Q heads, 2 KV heads, d_head 64, 32K
  ctx, RoPE θ=1e6; 12,288 B KV/token in fp16). Registry also has `smollm2-360m-instruct`,
  `qwen3-0.6b`, and `tiny-random` (2-layer random Qwen2 for CPU tests).
- dtype: **fp16 on MPS** (measured 2026-08-16: prefill 1751 tok/s, decode 23 ms/tok at 2K vs
  668 tok/s / 50 ms bf16), fp32 on CPU/tests, bf16 on CUDA. NaN guards in debug mode.
- Custom attention `"kvrl"` registered via `AttentionInterface` (transformers 5.15):
  - fast path = `F.scaled_dot_product_attention` with our own lower-right causal mask
    (`key j visible to chunk query i iff j <= kv - q + i`); correct for prefill, decode,
    chunk-after-compaction, and per-layer ragged caches; 4-D `attention_mask` passthrough
    is the masked-reference hook.
  - stats path ("dual") = same SDPA output + a blocked softmax pass (query sub-blocks of
    32/64) that accumulates attention mass received per cache slot into
    `StatsBuffer[n_layers, max_slots]` (fp32). Output is bit-identical to the fast path.
- Manual generation loop (no `generate`): explicit absolute `position_ids` (HF 5.15 has
  no `cache_position` on Qwen2Model; RoPE keys are stored post-rotation, so compaction
  never disturbs positions), `logits_to_keep=1` for prefill chunks. Greedy reference =
  `generate(do_sample=False, repetition_penalty=1.0)`.
- Chunk sizes: prefill 256 (fast path), 64 when stats are on; controller decision every
  C = 64 tokens (prefill chunk boundary / every 64 decode steps).

## 3. Cache layer (kvrl/cache)

- `KVCacheView` wraps `DynamicCache`: per-layer `slot → absolute position` tables,
  per-slot metadata (insertion step, is_generated, K/V norms, adjacent-key cosine),
  running attention statistics compacted alongside the tensors.
- `compact(cache, keep_slots)`: `index_select(2, keep)` on every layer's keys/values;
  shared keep-set across layers in v1; per-layer keep-sets supported by construction (v3).
- `MaskedReference`: keeps the full cache and expresses eviction as a 4-D bool mask —
  the correctness oracle (physical eviction ≡ masking: 8.9e-8 fp32, 3.5e-2 fp16 real).

## 4. Controller contract (kvrl/controllers)

```python
class KVCacheController:
    def decide(self, state: CacheState, budget: int) -> torch.LongTensor  # sorted keep slots
```
`CacheState` (identical in sim and real): per-slot `position, age, is_generated,
attn_last (renormalised, layer-mean), attn_last_max (layer-max), cumulative stats,
k_norm, v_norm, adj_key_cos, chunk id`, plus globals (context length, step, budget,
phase, occupancy). Protected: S=4 sink slots + the current chunk (W=64), always kept.

## 5. RL (kvrl/features, kvrl/sim, kvrl/rl) — see ML_SPEC.md

- Decision every C=64 tokens; must evict m = n − B (hard budget); action = ordered evict
  set sampled by Plackett–Luce (Gumbel-top-k) over per-token scores; exact per-slot logp.
- Obs: 18 per-token + 8 global features from `FeatureState` (one code path for both envs).
- Reward (sim): r_k = −Σ_{j∈E_k} F^γ_k(j)/r_scale (future attention mass charged at eviction)
  + terminal task term (gold-span retained) on labelled episodes. Quality-only in v1.
- Policy v1: MLP 26→128→128→1 (20K params); value net pooled; PPO with per-slot ratios,
  RB entropy; PPO hparams in ML_SPEC §9.
- Env A `CacheSimEnv`: batched trace replay (numpy/torch), millions of transitions/hour.
- Env B `RealCacheEnv`: same observation/action contract over the live model.

## 6. Traces (kvrl/traces) — see DATA_SPEC.md

Per prompt: token ids, positions, per-chunk × per-key attention mass (layer-mean and
layer-max, fp16, lower-triangular), per-token K/V norms and adjacent-key cosine (layer-mean),
task labels (critical token mask), full-cache greedy continuation. Stored as npz (+ parquet
index). Target ≤ 1.5 GB total on this disk.

## 7. Evaluation (kvrl/eval) & benchmarking (kvrl/bench)

Tasks: needle, kv, multihop, dependency, code (this repo), lm (NLL of true continuation);
metrics: accuracy, ΔNLL, output fidelity vs full cache, KV bytes, peak memory, latency
breakdown (prefill / decode / controller / cache ops / stats), throughput. Statistics:
paired bootstrap CIs per prompt. Tracker: `runs/<id>/`.

## 8. Product (kvrl/server + frontend) — see PRODUCT.md

FastAPI over `runs/` + live demo endpoint; React/Vite/TS dashboard (overview, token
retention strip, decision trace, experiment explorer, Pareto frontier); `demo.py` terminal demo.

## 9. Phase plan (vertical slice first)

P1 models+cache+bench+correctness → P2 traces+sim → P3 baselines → P4 PPO → P5 real
integration → P6 representation → P7 hierarchy → P8 hardware-aware → P9 suite → P10
dashboard → P11 perf → P12 docs/demo.
