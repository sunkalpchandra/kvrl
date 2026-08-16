# STATUS  (update after every significant piece of work)

_Last updated: 2026-08-16 16:40 — v1 complete + issue-driven second pass (E-012…E-017); final policy = ppo_mlp_v1; dashboard live; CI green_

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

## Results so far (see EXPERIMENTS.md, docs/results.md)
- Traces: 99 train / 20 val (70 MB). Regressor baseline (val corr 0.89). PPO v1 = warm start +
  layer-max reward, 60K steps (E-006). E-proxy: layer-max proxy ρ=0.59 vs real ΔNLL (E-004b).
- Real evaluation (E-007): RL beats h2o/snapkv/window/random at all budgets (acc + NLL; paired
  NLL vs h2o significant at 25/50%) but the key-norm heuristic is better on this suite
  (needle-driven). Failure analysis: RL relies most on key_norm; keeps critical tokens far
  better than attention heuristics but less than keynorm.
- v1.1 (λ_task=3) negative (E-008).

- Long-context eval (E-010): at 8K/16K RL answers needles at 50% (3/3) and 25% (2/3) budget
  while h2o/snapkv/window score 0; NLL within 0.02–0.12 of full cache. Policy trained ≤4K.
- Benchmark (bench run 20260816-045836): KV peak = budget + chunk exactly; decode 28–57 ms/tok
  up to 8K full cache, memory cliff at 16K (1.7 s/tok, swap); stats-path controllers ~1.5–2×
  prefill vs plain; controller overhead ≤2% of model time.

- Benchmark v3 (E-011, after the MPS pool-release fix D-010/BUG-003): 8K full-cache decode
  58 ms/tok (was 1.7–1.8 s), prefill 14 s (was 58–65 s); curve 28→88 ms/tok 512→8K, 1.09 s/tok
  at 16K (machine cliff).

## Second pass (addressing the caveats)
- Eval v2 (E-017): 42 prompts / 924 runs, ceiling tasks redesigned (kv 16 records: full 0.75;
  dependency copy-chain), regressor + two RL variants. RL v1 significantly beats h2o/snapkv/
  regressor on accuracy+fidelity and window on fidelity; vs keynorm not significant.
- NLL semantics fixed: only lm rows are quality NLL (graded rows' nll = answer confidence).
- Credit assignment work: per-slot counterfactual advantage (v1.3) moved sim metrics a lot but
  regressed real accuracy → documented sim-objective/task mismatch; v1 stays primary.
- Harness bugs fixed: MPS allocator bloat (BUG-003), deepcopy in the decode curve (BUG-004).
- Benchmark v4 (5 repeats, keynorm + rl) is the canonical latency table.

## In progress
- Ablations with the warm-start protocol (scripts/ablate.py, 20K steps) — resumed after the
  benchmark; results land in docs/ablations.md when done.

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
