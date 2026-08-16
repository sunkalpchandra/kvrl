# STATUS  (update after every significant piece of work)

_Last updated: 2026-08-16 02:45 (Phase 4 in progress: traces collecting, PPO pipeline verified)_

## Works right now (all verified by tests or real runs)
- **Model stack**: `kvrl.models` — Qwen2.5-0.5B-Instruct on MPS fp16 (also CPU/CUDA), registered
  `kvrl` attention (own causal mask, `enable_gqa` SDPA, dual stats path). Full-cache engine output
  == HF `generate(do_sample=False, repetition_penalty=1.0)` on the real model (verified).
- **Cache**: `KVCacheView` (CPU metadata, lazy K/V norms), physical eviction via `index_select`,
  masked-reference oracle; eviction ≡ masking test passes for 5 controllers (fp32 tiny model).
- **Engine**: chunked prefill (C=64) → controller decision → compaction → decode; timings split
  into prefill/decode/controller/compact; alive mask + per-decision records; teacher-forced NLL.
- **Controllers**: full, window, random, snapkv, h2o, tova, keynorm, hybrid, rl, regressor, oracle(sim).
- **Tasks**: needle, kv, multihop, dependency, code (this repo), lm — Gutenberg filler, seeded,
  critical spans; metrics (accuracy, ROUGE-L fidelity, prefix agreement, bootstrap CIs, paired).
- **Traces**: `kvrl.collect` (resumable) — ~0.15 MB per 2K trace, ~1 MB per 8K trace (npz).
- **Simulator**: `CacheSimEnv` replays traces with R1 future-mass reward; heuristics run unchanged;
  R1 ≡ delayed-lost-mass telescoping identity unit-tested.
- **RL**: PL/Gumbel-top-k sampler (exact per-slot logp, RB entropy — 9 tests), MLP/DeepSets/ISAB
  policies, PPO (per-slot clip, separate grad clipping), `kvrl.train`. Smoke run (20 traces, 8
  updates): deterministic RL beats H2O on sim lost mass at all budgets — NOT yet validated on real.
- **CLIs**: kvrl.collect / kvrl.train / kvrl.evaluate / kvrl.benchmark / demo.py.
- Tests: 57 passing (`pytest -q`), ruff clean.

## In progress
- Trace collection run (train ~96 + val 20 traces at 2K/4K/8K) — background.
- Next: full PPO training run → real-model evaluation (E-proxy + task suite) → benchmark → dashboard.

## Measured facts to remember
- MPS fp16 vs bf16: 2.6× faster (D-004). enable_gqa vs repeat_kv: 4–25× faster attention on MPS.
- Machine swaps heavily (Chrome/VS Code): latency noise is large; use warmup+repeats+medians;
  first compaction ~100 ms (kernel warmup).
- Stats path (dual) ≈ 3× the fast attention at 2K on MPS; trace collection ≈ 10 s per 2K prompt,
  ~60 s per 8K prompt.

## Known broken / risky
- Batch > 1 unsupported (custom attention drops HF padding masks) — v1 scope.
- Real-inference decode ~30 tok/s at 2K with controller (noise ±50%) — profile in Phase 11.
- code task at 2K produced no instance for one seed before the packing fix (now fixed; re-run collect to fill).
