# Inference Engineer design — HF integration, KV cache internals, attention stats

_Written 2026-08-16 from EMPIRICAL probes against the installed stack:
Python 3.12.0, torch 2.13.0 (MPS), transformers 5.15.0, Apple M2 8 GB.
Probe scripts (throwaway) live in the session scratchpad `probe/p1_cache.py … p8_mem.py`;
every snippet below was run and the outputs are pasted verbatim. Timings were taken
with only ~80 MB of free pages on the machine (VS Code etc. resident) and have large
IQRs — treat them as order-of-magnitude, re-measure in Phase 1 with the benchmark harness._

## 0. TL;DR facts (all verified on transformers 5.15.0 / torch 2.13.0)

1. `DynamicCache.layers[i].keys / .values` are plain tensors `[B, n_kv_heads, seq, head_dim]`
   (Qwen2.5-0.5B: `[1, 2, seq, 64]`, dtype = model dtype). Reassigning them is the eviction API.
2. `Qwen2Model.forward` has **no `cache_position` argument** in 5.15 (params:
   `input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, **kwargs`).
   Passing `cache_position=` is silently swallowed by `**kwargs`. RoPE uses **`position_ids` only**;
   default `position_ids = arange(q) + cache.get_seq_length()` — WRONG after compaction, so we
   must always pass explicit absolute `position_ids`.
3. A custom attention registered as `"kvrl"` receives `attention_mask=None` always (HF drops even
   2-D padding masks for names not in `AttentionMaskInterface`), unless we also register a mask
   function under `"kvrl"`. An explicit **4-D** mask (bool or float) is passed through untouched.
   `position_ids` arrives in the attention function's `**kwargs`; `module.layer_idx` is available.
4. Physical eviction (`index_select` on dim 2 of every layer) ≡ `-inf`/False masking of the same
   positions: max |Δlogits| = 8.9e-8 (fp32 tiny model), 3.5e-2 on logits of scale ~22 (real
   0.5B, fp16, MPS, same argmax). Chunked prefill ≡ one-shot: 2.4e-7 fp32.
5. Manual greedy loop ≡ `generate(do_sample=False)` token-for-token — **but only with
   `repetition_penalty=1.0`**: Qwen2.5-Instruct's `generation_config.json` ships
   `repetition_penalty: 1.1`, which `generate` applies even in greedy mode. Trap recorded.
6. MPS on M2 + torch 2.13 supports fp16 **and bf16** matmul/SDPA/index_select; SDPA accepts bool
   and float(-inf) masks (identical results); `recommended_max_memory()` = 5.33 GiB.
7. Stats capture cost: manual blocked attention is ~1.5–1.8x plain SDPA on CPU fp32 and ~3–6x on
   MPS fp16 for a 64-token chunk; per-key stats need at most `H × qblock × kv` fp32 (28 MiB at
   chunk 64 / kv 8K). Recommendation: SDPA-only fast path by default; stats path opt-in.

## 1. Cache API (transformers 5.15.0)

```python
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM
cfg = Qwen2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  intermediate_size=128, vocab_size=1000, max_position_embeddings=4096); cfg._attn_implementation="sdpa"
m = Qwen2ForCausalLM(cfg).eval()
c0 = DynamicCache()             # lazy: layers appended on first update()  (len(layers)==0 until forward)
c1 = DynamicCache(config=cfg)   # eager: 2 DynamicLayer objects, is_initialized False, tensors None until first update
out = m(input_ids=ids10, past_key_values=c1, use_cache=True)   # returns the SAME cache object (out.past_key_values is c1)
```
Output (p1_cache.py):
```
no-config: len(layers)= 0 layer_class_to_replicate= <class 'transformers.cache_utils.DynamicLayer'>
config:    len(layers)= 2 types= ['DynamicLayer', 'DynamicLayer'] is_initialized: [False, False]
after 10 tokens: get_seq_length= 10 layer0 keys shape (1, 2, 10, 16) values (1, 2, 10, 16) torch.float32
attributes on layer: ['keys', 'values', 'is_initialized', 'dtype', 'device']
after +3 tokens: get_seq_length= 13 keys (1, 2, 13, 16)      # forward with past_key_values appends via torch.cat(dim=-2)
get_mask_sizes(q=3, layer 0): (16, 0) get_query_offset: 13   # kv_length = cache_len + q, kv_offset 0, q_offset = cache_len
after eviction: get_seq_length= 8 keys (1, 2, 8, 16)         # after index_select on every layer
after crop(-2): 6                                             # crop() exists; NEGATIVE = remove n tokens from the end
cache_position kwarg: ACCEPTED silently (goes to **kwargs)
iter yields: [(1, 2, 11, 16), (1, 2, 11, 16), None]           # iter(cache) -> (keys, values, sliding_tensor) per layer
per-layer lengths: [5, 11] Cache.get_seq_length() (layer 0): 5 layer1: 11   # ragged per-layer lengths are tolerated
```
Facts:
- `DynamicLayer.update` = `torch.cat([self.keys, key_states], dim=-2)` (no preallocation → O(n) copy per
  step; fine for ≤32K on this laptop, but it means eviction should be batched, not per-token).
- Physical eviction of arbitrary positions (the ONLY thing we need):
  ```python
  keep = torch.tensor([...], device=layer.keys.device)          # sorted kept cache slots (not absolute positions!)
  for layer in cache.layers:
      layer.keys   = layer.keys.index_select(2, keep)
      layer.values = layer.values.index_select(2, keep)
  ```
  Existing helpers: `crop(-n)` (suffix only), `batch_select_indices` / `reorder_cache` (batch dim only),
  `reset()`. There is no built-in arbitrary-position removal → we own it in `kvrl/cache`.
- `DynamicCache(config=model.config)` is preferred (creates the right layer classes; Qwen2.5-0.5B has
  `layer_types = {'full_attention'}`, `use_sliding_window=False`, so all layers are `DynamicLayer`).
- Cache is stateful and mutated in place; K stored **after RoPE** (rotation applied in
  `Qwen2Attention.forward` before `past_key_values.update`), so evicting does not disturb the
  remaining keys' positions — new tokens just need their true absolute `position_ids`.
- Follow-up forward after compaction: pass `position_ids=torch.arange(P, P+q)[None]` where P is the
  true absolute position of the first new token. Do NOT pass `cache_position` (ignored). Verified control:
  omitting `position_ids` after evicting 9 of 64 → RoPE at 55.. instead of 64.. → |Δlogits| 5.5e-3 on
  the tiny model (vs 1.8e-7 when correct).

## 2. AttentionInterface + masks

Registration and call signature (p2_attn_iface.py):
```python
from transformers import AttentionInterface
def kvrl_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    # query [B,H,q,d]; key/value [B,Hkv,kv,d] (ALREADY concatenated with the cache); must return
    # (attn_output [B,q,H,d]  i.e. .transpose(1,2).contiguous(),  attn_weights or None)
    ...
AttentionInterface.register("kvrl", kvrl_attention)          # global; also visible in ALL_ATTENTION_FUNCTIONS
model.set_attn_implementation("kvrl")   # OK in 5.15;  or  model.config._attn_implementation = "kvrl";
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, attn_implementation="kvrl")   # also OK
```
Output:
```
call record layer0: {'layer_idx': 0, 'q': (1, 4, 12, 16), 'k': (1, 2, 12, 16), 'v': (1, 2, 12, 16), 'mask': None,
                     'kwargs': ['position_ids', 'sliding_window', 'use_cache'], 'scaling': 0.25, 'is_causal': None, 'n_kv_groups': 2}
kvrl vs sdpa logits max abs diff (prefill, no cache): 0.0
prefill 8 (cache empty) -> None   chunk 3 (cache 8) -> None   decode 1 (cache 11) -> None
decode 1 with 2-D padding mask, custom name -> None            # <-- padding masks are DROPPED for unregistered names
explicit 4-D bool mask, custom name -> ((1, 1, 1, 14), torch.bool)   (passed through untouched)
explicit 4-D float mask, custom name -> ((1, 1, 1, 15), torch.float32)
[mask registered=sdpa_mask] chunk 3 (cache 8) -> ((1, 1, 3, 11), torch.bool); prefill 8 -> None; decode 1 -> None
sdpa_mask(batch=1, q=3, kv=11, q_offset=8): [[1 1 1 1 1 1 1 1 1 0 0],[1 1 1 1 1 1 1 1 1 1 0],[1 1 1 1 1 1 1 1 1 1 1]]
```
Mechanism (masking_utils.py `_preprocess_mask_arguments`): 4-D tensor → returned as-is;
`config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping` → mask `None`;
otherwise `sdpa_mask` builds `[B,1,q,kv]` bool with `q_offset = cache.get_query_offset() = cache_len`
and returns `None` when `q==1` or `kv==q` (SDPA is_causal skip). Note the HF mask uses *cache length* as
the query offset, so it is structurally correct even after compaction (all cached keys visible + causal
inside the chunk) — but relying on it couples us to `is_causal`-skip semantics.

**Decision — safest mask strategy:** do NOT register a mask function. Inside `kvrl_attention` build our
own mask from shapes only ("lower-right aligned causal"): key slot `j` visible to chunk query `i` iff
`j <= (kv - q) + i`. This is correct for prefill (`kv==q`), decode (`q==1`), chunk-after-cache and
chunk-after-*compacted*-cache, and per-layer ragged caches (kv differs per layer, mask built per call).
If a 4-D `attention_mask` is given, use it verbatim (this is how the masked reference is expressed).
Batch>1 with padding is out of scope for v1 (would need our own padding handling); assert `B==1` or
no padding at the boundary. GQA: use `repeat_kv` (expand+reshape); `enable_gqa=True` also works with a
mask on CPU and MPS in torch 2.13 (diff 0.0) but HF only enables it when mask is None; keep `repeat_kv`
for the manual/stats path (avoid broadcast 5-D matmuls — measured 98–151 ms at kv 8K, pathological).

## 3. Per-key attention statistics with bounded memory

Reference kernel (p3b_stats.py; `row` = `stats[layer_idx]`, a preallocated `[n_layers, max_len]` fp32
buffer indexed by *cache slot*, mapping slot→absolute position kept by the cache view):
```python
def manual(q, k, v, scaling, addmask, row, qb, sm_dtype=torch.float32):
    G = q.shape[1] // k.shape[1]; kk = repeat_kv(k, G); vv = repeat_kv(v, G)
    B, H, Q, d = q.shape; kv = k.shape[2]; out = torch.empty_like(q)
    for s in range(0, Q, qb):                                     # query sub-blocks: peak = B*H*qb*kv fp32
        e = min(s + qb, Q)
        sc = torch.matmul(q[:, :, s:e], kk.transpose(2, 3)) * scaling + addmask[:, :, s:e]   # additive 0/-inf [1,1,Q,kv]
        p = torch.softmax(sc, dim=-1, dtype=sm_dtype)
        row[:kv] += p.sum(dim=(0, 1, 2)).float()                  # attention mass received per key (sum heads+queries)
        out[:, :, s:e] = torch.matmul(p.to(q.dtype), vv)
    return out            # exact softmax attention; vs SDPA: 4e-7 (fp32 CPU), 6e-5..4e-4 (fp16 MPS)
```
Sanity: total accumulated mass == H×Q (896.0 for H=14, Q=64) at every kv. Peak transient = `H×qb×kv×4 B`
= 7 / 14 / 28 / 56 MiB for qb=32 at kv 2K/4K/8K/16K (qb=64 doubles it) — never the O(n²) map.

Measured (B=1, H=14, Hkv=2, d=64, real Qwen-0.5B dims; median ms per layer-call; noisy machine):
```
CPU fp32, q-chunk 64:      kv=2048  sdpa 3.3 | manual qb64 5.9 | manual qb32 6.4 | dual(sdpa+stats) 8.4
                           kv=8192  sdpa 34  | manual qb64 52  | manual qb32 89  | dual 132   (IQRs up to 2x)
MPS fp16, q-chunk 64:      kv=2048  sdpa 2.2 | manual qb64 9.6 | qb32 11.9 | dual 17.9
                           kv=8192  sdpa 8.3 | manual qb64 55  | qb32 130  | dual 50
MPS fp16, decode q=1:      kv=8192  sdpa 4.9 | manual 20  | dual 15         kv=2048: 3.3 vs 2.8 (noise level)
MPS fp16 sdpa + stats from last nq of 64 queries: kv=8192 nq=8 9.3 | nq=16 9.6 | nq=64 17.0  (sdpa 4.75)
```
Reading: on MPS the fused SDPA kernel is 3–6x faster than the unfused manual path; the "dual" variant
(exact SDPA output + separate blocked softmax for stats) costs about the same as manual-only at long kv
because QK^T is done twice but PV once, and its output is bit-identical to the plain path.
**Recommendation:**
- Two modes in the same registered function, selected by a module-level/controller flag:
  `stats=False` → plain SDPA (identical to HF numerics, fastest); `stats=True` → **dual** (SDPA output
  + blocked stats pass, qb=32 on CPU, 64 on MPS). Dual keeps the *output* identical to the fast path,
  so switching stats on/off cannot change generated tokens (important for honest benchmarks).
- Extrapolated cost of stats over a full 8K prefill (128 chunks × 24 layers, kv growing 0→8K, mean
  ~half of the kv=8K numbers): ~+30 s CPU-fp32 / ~+40 s MPS-fp16 per 8K document → acceptable for
  offline trace collection (Phase 2), NOT for latency benchmarks (report latency with stats off, and
  separately "controller-input cost" with stats on).
- Cheaper decode-time proxy: stats from the last `nq=8..16` queries of a chunk costs ~2x SDPA instead
  of ~4x; the H2O-style cumulative score can be estimated from every decode step (q=1) at ~2–4x SDPA.
  Which one the ML side wants is an open question for reconciliation (see §7).

## 4. Correctness harness (verified numbers)

p4_correctness.py (tiny random Qwen2, fp32, CPU; `kvrl` = own-mask SDPA function from §2):
```
[sdpa] chunked(16) prefill vs one-shot: max|diff|=2.384e-07  allclose(atol=1e-5)=True
[kvrl] chunked(16) prefill vs one-shot: max|diff|=2.384e-07  allclose(atol=1e-5)=True
[eager] chunked(16) prefill vs one-shot: max|diff|=2.086e-07  allclose(atol=1e-5)=True
manual greedy == generate(do_sample=False): True   |   generate(kvrl) == generate(sdpa): True
eviction (index_select) vs masked(-inf) reference over 5 decode steps: max|dlogits|=8.941e-08 allclose(atol=1e-5)=True same argmax=True
chunk-of-4 after eviction vs bool-masked reference: max|dlogits|=1.788e-07
control (default position_ids after compaction => RoPE at 55.. instead of 64..): max|dlogits| = 5.504e-03
```
Real model, Qwen/Qwen2.5-0.5B-Instruct, fp16, MPS (p6_real.py / p6b_real.py):
```
loaded on mps float16 in 4.8s; alloc=942MiB driver=1040MiB; layers 24 heads 14 kv_heads 2 head_dim 64; params 494.0M
generate(sdpa) == generate(kvrl): True
manual loop == generate(do_sample=False): False            # generation_config.json has repetition_penalty 1.1 !
manual loop == generate(do_sample=False, repetition_penalty=1.0): True   (24 tokens, 410-token prompt)
REAL eviction(index_select, 205 of 410) vs bool-masked reference, 5 steps, fp16: max|dlogits|=3.516e-02 same-argmax=True (logit scale ~22)
REAL fp16 chunked(128) vs one-shot last-token logits: max|diff|=9.766e-02, same argmax=True
REAL mps fp16 prefill 2048 in chunks of 256: 3.64s (562 tok/s); decode @2K: median 127.6 ms/token; alloc=1021MiB driver=2195MiB
```
Harness definition for `tests/`:
- **T1 budget=100% ≡ HF greedy**: `generate(do_sample=False, repetition_penalty=1.0, temperature=None,
  top_p=None, top_k=None)` vs our loop; exact token match (fp32 CPU tiny model in CI; real model `-m slow`).
- **T2 chunked prefill ≡ one-shot**: fp32 atol 1e-5 (achieved 2.4e-7); fp16 real: compare argmax + atol 0.2.
- **T3 eviction ≡ masking**: build 4-D bool mask `[1,1,q,cache_len+q]` False at evicted slots and above
  the diagonal inside the chunk (mask on the *full* cache); fp32 atol 1e-5 (achieved 8.9e-8); fp16 real:
  atol 0.1 + same argmax over ≥5 steps.  Masked reference must NEVER produce a fully masked row (current
  token always attends to itself); torch 2.13 SDPA returns 0 (not NaN) for fully masked rows on CPU+MPS anyway.
- **T4 position control**: assert that the "wrong positions" variant differs (guards against a future
  HF change silently making `position_ids` unused).

## 5. MPS specifics (torch 2.13.0, Apple M2)

p5_mps.py output:
```
recommended_max_memory: 5.333 GiB      (== the "≈5.7 GB" in CLAUDE.md, GiB vs GB)
float16 / bfloat16 / float32: matmul OK, sdpa(no mask) OK, sdpa(bool mask) OK (bool vs float(-inf) mask diff 0.0), index_select OK
fully-masked query row -> cpu: [0,0,0] mps: [0,0,0] (no NaN)
alloc delta for 48 MiB fp16 tensor: current 48.0 MiB, driver 1024.0 MiB   # driver pool grows in 1 GiB steps
after del + empty_cache: current 0.1 MiB, driver 10.6 MiB
has torch.mps.Event: True | has torch.mps.profiler: True; enable_gqa (no mask) on mps: OK
```
Helpers (to live in `kvrl/bench/measure.py`):
```python
def sync(device): {"mps": torch.mps.synchronize, "cuda": torch.cuda.synchronize}.get(device.type, lambda: None)()
def mem(device):   # bytes: (allocated_by_tensors, reserved_by_driver)  + analytic KV bytes reported separately
    if device.type == "mps":  sync(device); return torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()
    if device.type == "cuda": return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    return 0, 0
def timeit(fn, device, n=10, warm=3): warm-up, then per-iter sync→perf_counter→fn→sync; report median + IQR
```
Notes: `driver_allocated_memory` is what the OS sees (1 GiB granularity, includes cached pool) — report
both; `current_allocated_memory` is the honest tensor footprint. dtype choice: fp16 works and matched
HF; bf16 also works on this M2 build — bf16 has more range (Qwen was trained bf16) but MPS bf16
performance was not measured (open question). Default fp16 with an `--dtype` flag; keep fp32 for tests.

## 6. Memory arithmetic (Qwen2.5-0.5B: 24 layers, 2 kv heads, head_dim 64)

KV bytes/token fp16 = 24 × 2 (K,V) × 2 heads × 64 × 2 B = **12,288 B** (fp32: 24,576).
```
 tokens | KV fp16 | KV fp32 | fp32 score block chunk64 (H=14) | chunk256 | fp16 logits if one-shot prefill (V=151936)
   2048 |  24 MiB |  48 MiB |   7 MiB  |  28 MiB | 0.58 GiB
   4096 |  48 MiB |  96 MiB |  14 MiB  |  56 MiB | 1.16 GiB
   8192 |  96 MiB | 192 MiB |  28 MiB  | 112 MiB | 2.32 GiB
  16384 | 192 MiB | 384 MiB |  56 MiB  | 224 MiB | 4.64 GiB
  32768 | 384 MiB | 768 MiB | 112 MiB  | 448 MiB | 9.27 GiB
```
Weights: 494.0 M params → 942 MiB fp16 (measured `current_allocated` after load). The KV cache is NOT the
memory problem at ≤32K on this model; the transients are: (a) full-vocab logits — always call the model
with `logits_to_keep=1` for prefill chunks (verified: `logits.shape == (1,1,V)`), (b) SDPA's math
fallback on MPS/CPU materialises `H×q×kv` scores in model dtype (`~58 MiB` at chunk 256 / kv 8K fp16),
(c) our stats block. Budget on 8 GB with 5.33 GiB recommended max: 0.92 GiB weights + ≤0.4 GiB KV
+ ≤0.5 GiB transients + OS/other apps → **16K is comfortable, 32K feasible only in fp16 with chunk ≤256
and stats off; the practical limit is time** (prefill 562 tok/s at 2K on MPS, quadratic growth).
Default chunk sizes: prefill 256 (MPS fast path), 128 (CPU), **64 when stats are captured** (bounded
28 MiB score block at 8K), decode q=1. Eviction batched every chunk (prefill) / every k tokens (decode).

## 7. Module structure and contracts (proposal for reconciliation)

```
kvrl/models/base_model.py     BaseCausalLM(ABC): load(name, dtype, device) ; forward_chunk(ids, position_ids, cache, logits_to_keep) ;
                              prefill(ids, chunk, cache, controller=None) ; decode_step(token, pos, cache) ;
                              greedy_reference(ids, n)  (= HF generate w/ repetition_penalty=1.0) ; kv_bytes_per_token ; n_layers/n_kv_heads/head_dim
kvrl/models/hf_model.py       HFCausalLM(BaseCausalLM): transformers 5.x impl; registers "kvrl" attention ONCE at import;
                              compat layer (asserts: DynamicCache has .layers[i].keys, forward has no cache_position, ...) fails loudly on drift
kvrl/models/attention.py      kvrl_attention(module, q, k, v, mask, ...): fast SDPA path | dual stats path; own lower-right causal mask;
                              writes into StatsBuffer (thread-local/module-global handle set by HFCausalLM before each forward)
kvrl/models/model_registry.py name -> (hf_id, dtype default, ctx max, chat template flag); "qwen2.5-0.5b-instruct" first; "tiny-random" for tests
kvrl/cache/view.py            KVCacheView: wraps DynamicCache; per-layer slot->absolute-position tables (LongTensor per layer),
                              per-slot metadata (position, insertion step, K-norm/V-norm cached), get_state() -> CacheState
kvrl/cache/compact.py         evict(cache, keep_slots_per_layer | shared keep_slots) via index_select on every layer + updates tables + stats buffer compaction
kvrl/cache/reference.py       MaskedReferenceCache: keeps the FULL cache, materialises 4-D bool mask from the evicted-position set (test oracle only)
kvrl/cache/stats.py           StatsBuffer: [n_layers, max_slots] fp32 attention mass this chunk + cumulative; K/V norms per layer per slot
```
`KVCacheController.decide(cache_state, attention_state, budget) -> keep_slots (per layer or shared)`.
What the inference side can supply **cheaply per cache slot** (all O(L) per layer, no O(n²)):
- `position` (absolute), `age = current_pos - position`, `is_sink` (first k), `layer_len`;
- `attn_mass_chunk[layer, slot]` = attention received during the last chunk / decode step (from the stats
  buffer, summed over heads and chunk queries; optionally per-head-group `[layer, Hkv, slot]` at 2x cost);
- `attn_mass_cum[layer, slot]` (running sum, compacted alongside the cache — H2O's statistic);
- `k_norm[layer, slot]`, `v_norm[layer, slot]` (computed once when the token enters the cache, from
  `layer.keys[..., slot, :].norm(dim=-1)`; cheap; keep fp32);
- `budget` = target slots (per layer or total), `n_new` tokens about to be appended.
Not cheap / not offered in v1: full attention rows, per-head-per-query maps, value-weighted scores.

Per-layer eviction (different keep-sets per layer) — verified feasible (p7_ragged.py): the model runs
with ragged per-layer cache lengths (`[27, 32]`) under the custom attention because the mask is built
per call from that layer's `kv`; result vs a per-layer masked reference: **1.192e-07**. With the stock
`sdpa` implementation this breaks for q>1 (`RuntimeError: size of tensor a (42) must match ... (37)`),
because HF sizes the mask from layer 0. Consequences: `Cache.get_seq_length()` becomes per-layer
(pass `layer_idx`), the reference oracle must be a per-layer 4-D mask (cannot use the model's single
`attention_mask` arg → implement as a second registered attention fn `"kvrl_ref"` reading masks by
`module.layer_idx`), and slot→position tables must be per layer.

## 8. Open questions / not verified

- MPS bf16 speed vs fp16 (both work; only fp16 timed). Whether fp16 overflow occurs in Qwen2.5 activations
  at long context (bf16 is the model's native dtype) — check on the first 8K real run.
- Timings above were taken under memory pressure; re-measure with the harness before quoting.
- Stats granularity the ML/RL side needs (per layer, per kv-head, cumulative vs per-chunk, sampled queries) —
  decides between `dual` (~4x SDPA when on) and sampled (~2x).
- Batch>1 / padding with the custom attention (HF drops 2-D masks for our name): out of scope v1; would
  need `AttentionMaskInterface.register("kvrl", sdpa_mask)` + `is_causal`-skip handling.
- torch.compile of the decode step on MPS, `torch.mps.profiler` for kernel-level attribution — untested.
- Whether HF 5.16+ keeps `layers[i].keys` (5.15 already deprecates `get_max_cache_shape`); the compat
  layer must assert the shape contract at import and the test suite pins `transformers==5.15.*`.
