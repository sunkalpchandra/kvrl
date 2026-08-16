# DECISIONS  (append-only architectural decision log; newest at bottom)

Format: `D-NNN (date) — title` / Context / Decision / Consequences / Alternatives.

## D-001 (2026-08-16) — Project lives at ~/kvrl, package `kvrl`, repo sunkalpchandra/kvrl
Context: master build prompt names the CLI `python -m kvrl.*`.
Decision: package + repo both `kvrl`; frontend under `frontend/`.
Consequences: imports are `from kvrl...`; entrypoints `kvrl.train/evaluate/benchmark/demo`.

## D-002 (2026-08-16) — venv with `--system-site-packages`
Context: only ~9.5 GB free disk; torch 2.13 already in system Python 3.12; sibling
projects (gpu-optimizer) use the same pattern.
Decision: `.venv` reuses system torch/numpy/gymnasium; extra deps installed inside.
Consequences: `requirements.txt` still pins everything for clean machines; setup.sh
supports both modes.

## D-003 (2026-08-16) — Simulator-first RL; real inference for validation
Context: 8 GB laptop; running the LLM inside the RL loop is infeasible for millions
of transitions.
Decision: Env A (trace-replay simulator) trains; Env B (real HF inference with real
cache manipulation) validates and (late phase) fine-tunes.

## D-004 (2026-08-16) — Model: Qwen/Qwen2.5-0.5B-Instruct primary; fp16 on MPS
Context: 8 GB laptop; measured MPS fp16 prefill 1751 tok/s & decode 23 ms/tok vs bf16
668 tok/s & 50 ms/tok (2K ctx, 2026-08-16); model is bf16-native so fp16 overflow is a
risk to test at 8K. Registry keeps smollm2-360m, qwen3-0.6b, tiny-random.
Decision: default dtype fp16 on MPS, fp32 CPU, bf16 CUDA; `--dtype` override.

## D-005 (2026-08-16) — Custom "kvrl" attention with own causal mask; explicit position_ids
Context (verified, transformers 5.15): custom attention names get attention_mask=None;
Qwen2Model has no cache_position; RoPE uses position_ids only. 4-D masks pass through.
Decision: build lower-right causal mask from (q, kv) inside the attention fn; always pass
absolute position_ids; 4-D mask passthrough is the masked-reference oracle. Batch=1 only in v1.

## D-006 (2026-08-16) — Attention stats via "dual" path; fast path default
Context: manual blocked attention is 3–6× slower than fused SDPA on MPS.
Decision: stats off ⇒ plain SDPA; stats on ⇒ SDPA output + separate blocked softmax mass
accumulation (identical outputs). Latency benchmarks report stats-off model latency and
stats/controller cost separately and summed.

## D-007 (2026-08-16) — RL v1: hard budget, PL/Gumbel-top-k set action, R1 future-mass reward
Context: see design_ml_architect.md §2, §4. Bernoulli+projection rejected (biased ratio).
Decision: adopt the Recommended v1 spec (ML_SPEC.md). Sinks S=4 + current chunk protected.

## D-008 (2026-08-16) — Greedy reference must use repetition_penalty=1.0
Context: Qwen2.5-Instruct generation_config ships repetition_penalty=1.1, applied even in
greedy mode; our manual loop matches generate() only with penalty 1.0.
Decision: `greedy_reference()` sets repetition_penalty=1.0; tests assert token equality.

## D-009 (2026-08-16) — Reward uses layer-MAX lost attention mass; warm start from regressor
Context: E-proxy (E-004) shows the layer-mean proxy correlates weakly with real ΔNLL (ρ 0.44)
while the layer-max variant reaches 0.62 (per-budget 0.57–0.72) — retrieval heads are hidden by
averaging, as the ML Architect flagged. From-scratch PPO (E-003) stayed near-uniform and below
H2O in 16K steps.
Decision: `sim.layer_max_reward: true` by default; `rl.init_from: checkpoints/regressor_v1.pt`
warm start by default; the from-scratch/layer-mean runs remain as recorded ablations.
Consequences: reward scale recalibrated (r_scale from Amax); features unchanged.

## D-010 (2026-08-16) — Release the MPS allocator pool during chunked prefill
Context: BUG-003 — reserved memory balloons to 7.7 GB after an 8K chunked prefill on MPS;
the machine swaps and decode crawls (1.2 s/token) although only 0.9 GB is allocated.
Decision: `InferenceEngine` calls `empty_cache()` every 8 prefill chunks and before decode on
MPS (no-op elsewhere). Measured: 8K full-cache decode 1185 → 57 ms/token, prefill unchanged.
Consequences: benchmark re-run (bench run 3) supersedes earlier 8K/16K full-cache numbers.
