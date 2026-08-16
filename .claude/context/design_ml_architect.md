# design_ml_architect.md — RL formulation for kvrl (v1 + upgrade path)

Author role: ML Architect. Status: Phase-0 independent design (to be reconciled with the
Inference Engineer's design and then distilled into `ML_SPEC.md`/`ARCHITECTURE.md`).
Model assumed: Qwen/Qwen2.5-0.5B-Instruct (24 layers, 14 Q heads, 2 KV heads, d_head 64,
12,288 B KV/token bf16). Chunk C = 64. All numbers below are design estimates unless
marked "measure"; nothing here is a result.

Honesty note up front: at 8K context the whole KV cache is 100 MB against ~1 GB of
weights, so absolute memory savings on this model are modest. What we measure is the
KV-byte ratio, decode-attention latency vs cache length, and quality at fixed budget —
the mechanism scales; the numbers on this laptop are small by construction.

## 0. Vocabulary

- P prompt tokens, G generated tokens, C = 64 chunk size, K = ⌈(P+G)/C⌉ chunks.
- B = hard token budget (absolute, per episode; sampled as a fraction of P: 12.5/25/50 %).
- A_k(j) = full-cache attention mass that query chunk k puts on key token j, averaged over
  the C queries, 14 heads (GQA: 7 Q heads per KV head, mean over all 14) and 24 layers, so
  Σ_j A_k(j) = 1 per chunk. Trace also stores A^max_k(j) = max over layers of the head-mean.
- R_k retained set after decision k; E_k tokens evicted at decision k; E_{<k} = ∪_{k'<k} E_{k'}.
- â_k(j) = A_k(j) / Σ_{j'∈R_k} A_k(j') — mass renormalised over the retained set (what a
  real evicted-cache model would approximately show; see §4).

## 1. MDP

Step (decision point). One decision after each chunk of C tokens has been written to the
cache: prefill chunk k (tokens (k−1)C..kC−1) or every C decode steps. Before the decision
the cache holds n = |R_{k−1}| + C tokens; the policy must evict m = max(0, n − B) so that
|R_k| ≤ B. Peak occupancy between decisions is B + C (reported as such; "budget B" means
post-decision occupancy). Steps with m = 0 (early prefill, k ≤ B/C) are no-ops: the sim
still updates features but records no action/reward (they carry no gradient).

Episode = one prompt + its full-cache greedy continuation (teacher-forced from the trace,
G ≤ 512). Terminates after the last decode chunk (or at EOS). Deterministic given the
trace and the action sequence; stochasticity comes only from the policy.

Horizon (decisions with m > 0, C = 64, G = 256, B = 25 % of P):

| P   | prefill decisions | with eviction | decode decisions | total steps | sub-actions (m=64) |
|-----|-------------------|---------------|------------------|-------------|--------------------|
| 8K  | 128               | 96            | 4                | 100         | 6.4K               |
| 16K | 256               | 192           | 4                | 196         | 12.5K              |
| 32K | 512               | 384           | 4                | 388         | 24.8K              |

General formula: eviction steps = (P − B)/C + G/C (so 116/228/452 at B = 12.5 %, 68/132/260
at B = 50 %, G = 256). Steady-state m = C = 64 exactly (each step adds C tokens to a full cache);
only the first eviction step can have m < 64.
Candidate set size at B = 2048: n_cand = B + C − S − W = 2048 + 64 − 4 − 64 = 2044.

Transition (sim): R_k = (R_{k−1} ∪ chunk_k) \ E_k; then chunk k+1's stats â_{k+1}(·) over
R_k ∪ chunk_{k+1} update the feature table (§3). Real inference (Env B): identical
bookkeeping, but stats come from the live model and the cache is physically compacted.

## 2. Action space and sampler

Per decision the policy emits one real score s_i ("eviction propensity") per candidate
token; the sampler turns scores into an evict set of exactly m tokens.

### 2.1 Recommended: Plackett–Luce (PL) via Gumbel-top-k (exact per-slot log-probs)

Sample: g_i = s_i + Gumbel(0,1) i.i.d.; the ordered evict list (i_1,…,i_m) = indices of the
m largest g (descending). Kool et al. 2019 ("Stochastic Beams…") show this is exactly
sequential sampling without replacement from softmax(s), i.e. Plackett–Luce.

Log-probability of the ordered list, vectorised and numerically stable in O(n + m):
- U = unpicked indices; Z_U = logsumexp_{l∈U} s_l (one scalar).
- p_j = s_{i_j}; T_j = logsumexp_{l=j..m} p_l (reverse logcumsumexp over the picked list).
- D_j = logaddexp(Z_U, T_j) (log-partition of the j-th conditional; the remaining set at
  slot j is U ∪ {i_j..i_m}).
- log π(i_1..i_m | s) = Σ_{j=1}^m (p_j − D_j); per-slot logp_j = p_j − D_j.
Masked (protected/padded) tokens get s = −inf and drop out of every LSE.

PPO surrogate: treat the m slots as sub-actions of an augmented MDP with a shared advantage
Â (exactly what token-level PPO does in RLHF): L = −(1/m) Σ_j min(ρ_j Â, clip(ρ_j,1±ε) Â),
ρ_j = exp(logp_j − logp_j^old). Per-slot clipping avoids the ratio of a 64-draw sequence
exploding/vanishing; approx-KL and clip-fraction are averaged over slots. The
sequence-level ratio Π_j ρ_j is the ablation (expected worse). Order is a latent the
environment ignores; the return is order-invariant, so the gradient is unbiased for the
induced set policy (higher variance than the intractable set-level ratio, acceptable).

Entropy for the bonus — Rao–Blackwellised chain-rule estimator, unbiased for the ordered
distribution, O(n + m): H_j = D_j − e^{Z_U − D_j} w_U − e^{T_j − D_j} v_j, with
w_U = Σ_{l∈U} e^{s_l − Z_U} s_l (softmax-weighted mean score over U, one scalar) and
v_j = Σ_{l≥j} e^{p_l − T_j} p_l (reverse cumulative over the picked list). Use
H̄ = (1/m) Σ_j H_j (max ≈ log n_cand ≈ 7.6 nats) so the coefficient is interpretable.
Cheap fallback: H_1 (first-slot categorical entropy). Not recommended: −log π (unbiased,
very high variance).

Deterministic eval policy: top-m by s. Report both stochastic and deterministic numbers.

Unit tests (sampler is the riskiest piece of code): exhaustive Σ π = 1 for n = 5, m = 2;
Gumbel-top-k frequencies vs PL probabilities (n = 6, m = 2, 10^5 draws, χ²); RB entropy vs
exact entropy for tiny n; grad-check logp; masking; batch/ragged padding.

### 2.2 Rejected: independent Bernoulli per token + budget projection

Sample a_i ~ Bern(σ(s_i)); count Σa_i ~ N(Σp, √Σp(1−p)) ≈ ±8 around 64 → the projection
(top-m by p to fix the count) fires almost every step, so the executed action ≠ the sampled
action and log π of the executed action is unavailable → biased ratio; the policy can also
push all p_i → 0 and let the projection do the work (degenerate). Conditional-Poisson
sampling repairs exactness (DP in O(n·m) ≈ 135K ops per decision, sequential in m) but
parametrises the same ranking as PL with more code and less standard tooling. Verdict: PL.
When m > n/2 (never at steady state in v1) sample the keep set instead — same math.

### 2.3 Protected tokens: hard-protected, outside the action set (v1 default)

- S = 4 attention-sink tokens (StreamingLLM, Xiao et al. 2023): sinks receive large
  content-free mass; evicting them is catastrophic and trivially learnable — protecting them
  removes an unnecessary safety hazard during exploration. Measure on Qwen2.5-0.5B: mean
  A(·) on positions 0..7 from far queries per layer; pick S = smallest s with < 1 % mass on
  positions ≥ s among the first 8 (experiment E-sink; default 4).
- W = C = 64: the current chunk is protected because its tokens have only within-chunk
  statistics (immature features) in both sim and real; they become candidates one step
  later. Recency beyond that is NOT protected — the policy must learn it from `age`/`rel_pos`
  (H2O's 50 % recent window is a baseline, not a rule). Ablate W ∈ {64, 128, 256, B/2}.
- Protected tokens count against B. Budget for learned decisions = B − S − W.

## 3. State / observation

Contract (both envs must produce it from the same `FeatureState` module — one code path):
`tok: f32[n_cand, 18]`, `glob: f32[8]`, `cand_idx: i64[n_cand]`, plus a mask for ragged
batches. Attention-derived quantities use the length-invariant transform
ϕ(a; n) = log1p(a·n) / log1p(N_max) (a·n = ratio to uniform attention over the n retained
keys; N_max = 32768), so "uniform" ≈ 0.07 at any context length. Norm features are
z-scored with corpus constants frozen into the checkpoint. Running obs-normalisation
(mean/std) is applied on top and frozen at eval.

### 3.1 Per-token features (v1, 18 dims; all computable incrementally in sim AND real)

| # | name | definition / update (per retained token j at chunk k) |
|---|------|--------------------------------------------------------|
| 1 | age_log | log1p(k − chunk(j)) / log1p(512) |
| 2 | rel_pos | pos(j) / (current context length) |
| 3 | pos_log | log1p(pos(j)) / log1p(32768) |
| 4 | is_generated | 1 if j was produced in decode, else 0 |
| 5 | attn_last | ϕ(â_k(j)) — mass from the most recent query chunk |
| 6 | attn_ema_fast | e ← 0.5·e + 0.5·ϕ(â_k(j)); init from within-chunk stat |
| 7 | attn_ema_slow | same with β = 0.9 |
| 8 | attn_mean | cumulative Σ_k' ϕ(â_k'(j)) / age (H2O heavy-hitter score, age-normalised) |
| 9 | attn_max | max_k' ϕ(â_k'(j)) |
| 10 | attn_lastmax_layer | ϕ(â^max_k(j)) — layer-max mass from the last chunk (retrieval-head signal) |
| 11 | attn_disp | log((A^max_k(j)+ε)/(A_k(j)+ε))/5, clipped — layer dispersion (proxy for head/layer entropy) |
| 12 | hit_rate | (#chunks where j in top-5 % of retained keys by â) / age |
| 13 | since_hit | log1p(chunks since last top-5 % hit) / log1p(512) |
| 14 | chunk_share | j's share of its own chunk's cumulative mass, ϕ-scaled — within-chunk rank |
| 15 | key_norm | z-scored L2 norm of j's key (layer-mean; RoPE preserves norm) — Devoto et al. 2024 |
| 16 | value_norm | z-scored L2 norm of j's value (layer-mean) — value-aware pruning, Guo et al. 2024 |
| 17 | adj_key_cos | cosine(key_j, key_{j−1}) layer-mean, static — redundancy proxy (optional; ablate) |
| 18 | nbr_evicted | fraction of j's original chunk already evicted — "context dissolving" |

Sim-only (NEVER an actor input; allowed for critic/oracle/analysis): future mass
F_k(j) = Σ_{k'>k} A_{k'}(j) and its discounted version; un-renormalised A_k on evicted
tokens; needle/gold-span labels; per-layer vectors if stored. v2 candidates: dynamic
redundancy (cosine to nearest retained neighbour; needs a 16-dim random projection of keys
from 3 representative layers stored in the trace, ~96 B/token), token-id repeat flag,
"top-1 % in ≥ r layers" count (needs an extra uint8 per (chunk, key)).

### 3.2 Global features (8 dims, both envs)

budget_frac = B/P; occupancy = n/B (≥ 1 at decision); evict_frac = m/n_cand;
ctx_log = log1p(ctx_len)/log1p(32768); step_frac = k/K (K known: P known, G = max_new_tokens);
phase (0 prefill / 1 decode); remaining_gen = (max_new_tokens − generated)/max_new_tokens;
chunk_entropy = H(â_k(·))/log n — how spread the current queries are over keys.
Sim-only critic extras (asymmetric actor–critic, Pinto et al. 2017; default ON, ablated):
committed future loss Σ_{j∈E_{<k}} F_k(j); retained future mass Σ_{j∈R} F_k(j);
gold-span-retained fraction. Cumulative lost mass so far is sim-only → not an actor input.

Feature identity between envs is a hard requirement: same class, same update code, only
the source of (â_k, key/value norms, positions) differs. Verified by replaying an Env-B
episode's raw stats through the sim FeatureState and asserting equality.

## 4. Reward

### 4.1 Quality proxy: lost attention mass

Delayed (observed) lost mass at chunk k: ℓ_k = Σ_{j∈E_{<k}} A_k(j) ∈ [0,1] — the fraction
of the full-cache model's attention (chunk-k queries, layer/head-averaged) that lands on
tokens no longer in cache. Why defensible: for one layer, with o = Σ_j α_j v_j and the
evicted-cache output o' = Σ_{j∈R} α_j v_j / (1−ℓ): o − o' = Σ_{j∈E} α_j v_j −
(ℓ/(1−ℓ)) Σ_{j∈R} α_j v_j, so ‖o − o'‖ ≤ ℓ·max‖v‖ + (ℓ/(1−ℓ))·(1−ℓ)·max‖v‖ = 2ℓ·max‖v‖ —
the attention-output perturbation is first-order in ℓ (this is the H2O
argument, Zhang et al. 2023; sparsity of α makes small ℓ achievable at small budgets).
SnapKV (Li et al. 2024) and Scissorhands (Liu et al. 2023) supply the empirical premise
that attention patterns persist, so past mass predicts future mass. Renormalisation is
already in the bound: after eviction the softmax redistributes the missing ℓ over R
(factor 1/(1−ℓ)); we therefore (a) feed the policy renormalised â (what real inference
sees) and (b) charge the raw A on evicted tokens (what was lost). Known weaknesses of the
proxy: layer-averaging can hide a few retrieval heads (mitigated by the terminal task
term and by logging the layer-max version ℓ^max_k); mass on sinks is content-free (moot:
sinks are protected); errors compound across layers/steps (only real validation tells).
Variants to log: value-weighted ℓ^v_k = Σ_{j∈E} A_k(j)·‖v_j‖/mean‖v‖ (VATP-style).

### 4.2 Reward used for training (sim)

Charge each eviction its own future at decision time (R1, default):
  r_k = −(1/r_scale) · Σ_{j∈E_k} F^γ_k(j),   F^γ_k(j) = Σ_{k'=k+1}^{K} γ^{k'−k} A_{k'}(j).
Identity: the discounted return from step 0 under R1 equals exactly the discounted return
under the delayed reward R2: r_k = −ℓ_{k+1}/r_scale (telescoping over k; unit test this).
From step t they differ only by the "committed" loss of earlier evictions, which is
action-independent noise → R1 gives the same optimal policy and policy gradient in
expectation with much lower variance. R1 needs the future (trace) so it is sim-only —
fine, training is sim-only; R2 is what a real-inference fine-tuning phase would use.
r_scale = mean |r_k| of the random policy at B = 25 % (measure once, store in config).

Terminal task term (labelled synthetic episodes only: needle, KV-retrieval, multi-hop):
  r_T += λ_task · (fraction of gold-span tokens retained when the question chunk is
  processed and throughout generation), λ_task = 1.0 (≈ 5–10 % of |return| under R1 at
  B = 25 % by design; retune after E-scale below). Keep labelled episodes ≤ 30 % of the
  batch so the policy is not tuned to synthetic markers; general-text NLL guards against
  "hoard rare tokens".

Memory/latency in v1: none. m is fixed by the hard budget, so KV bytes (B+C tokens peak)
and attention cost are constants of the run; a "bonus for evicting below budget" would
make m an action (that is v2) and cannot help quality in a replay sim. v1 reward is
quality-only: R = Σ_k γ^k r_k(R1) + γ^T r_T. Latency and memory are measured in Env B and
plotted per B; the Pareto frontier in v1 comes from sweeping B ∈ {12.5, 25, 50 %}.

### 4.3 What must be measured on the real model to validate the proxy (E-proxy)

For ~100 prompts × {random, window, H2O, regressor, RL, oracle} × B ∈ {12.5, 25, 50 %}:
(a) sim lost mass ℓ (replay the same eviction decisions in the sim); (b) real lost mass
via the masked-reference mode (full cache + eviction mask; one extra un-masked softmax
per layer gives the evicted-model queries' mass on masked keys); (c) ΔNLL of the
full-cache greedy continuation under the evicted cache (teacher-forced), and per-position
KL(p_full‖p_evict); (d) task accuracy on labelled tasks. Report Spearman ρ across
(prompt, policy, budget) for (a)↔(b), (a)↔(c), (b)↔(c), (a)↔(d), plus per-budget
calibration plots. Acceptance: ρ(a,c) ≥ 0.7 and monotone per budget; if it fails, switch
the reward to ℓ^max or ℓ^v (whichever correlates better) before any RL run at scale.
Also measure feature drift: per-feature correlation sim vs real along the same
trajectory and Jaccard overlap of the evict sets chosen by the policy from sim features
vs real features (transfer diagnostic).

## 5. Policy / value architecture

v1 policy (MLP, per-token, set-agnostic): input [tok(18) ⊕ glob(8)] = 26 →
Linear 128 → SiLU → Linear 128 → SiLU → Linear 1 = score. Params 3,456 + 16,512 + 129 =
20,097. Cost at n_cand = 2044: ≈ 40 M MACs → ~1–3 ms on M2 CPU (torch, multithread) or
≈ 0.5 ms compute + ~1 ms launch overhead on MPS; feature update O(n) < 0.5 ms;
Gumbel-top-k negligible; observation is 2044 × 26 × 4 B = 213 KB (transfer trivial).
Decisions occur once per 64 decode steps → amortised ≈ 0.05 ms per generated token vs a
20–40 ms decode step (≈ 0.1–0.2 %); during prefill ≈ 2–5 % of a 64-token chunk forward.
Cache compaction (index_select over 24 layers, ~50 MB at B = 2048) is the larger cost and
belongs to the Inference Engineer's latency breakdown.

v1 value net (separate parameters; tiny nets, avoid interference): per-token
Linear 26→128 → SiLU, mean-pool ⊕ max-pool (256) ⊕ glob(8) ⊕ privileged(3) → 128 → 1;
≈ 3.5K + 34K + 129 ≈ 38K params.

v1.5 (cheap set-awareness, DeepSets): φ: 26→128→64 per token; pool mean⊕max → ctx (128) ⊕
glob → 64; ψ: [φ_i ⊕ ctx] 128→128→1. ≈ 37K params, ~2× v1 cost. Gives every token a view of
the score distribution of its competitors (relative ranking) without O(n²).

v2 (set-attention): token embed 26→64; 2 ISAB blocks (Set Transformer, Lee et al. 2019)
with 32 inducing points, 4 heads, d = 64, FFN 128, plus one global token carrying glob and
emitting the value; per-token linear score head. ≈ 75K params; ≈ n × 80K MACs ≈ 165 M
MACs at n = 2044 → ~5–15 ms CPU / ~2–3 ms MPS (still ≪ one decode step when amortised
over 64 tokens). Full O(n²) self-attention (≈ 640 M MACs) is not needed.

Warm start (recommended in addition to from-scratch): initialise the score head from the
supervised regressor of §6 (same MLP predicting −F^γ), then fit the value net for ~20
updates with the policy frozen, then PPO. The from-scratch run remains the clean test.

## 6. Baselines the policy must beat (all under identical S, W, B, C)

full cache · random · sliding window / StreamingLLM (S sinks + most recent) · last-chunk
attention (SnapKV-style observation window, top by â_k) · H2O (cumulative attention
heavy-hitters, B/2 recent + B/2 heavy; ratio configurable) · TOVA-like (evict min
last-query attention) · key-norm heuristic (evict largest ‖k‖) · recency+attention hybrid.

Honest ML baseline: supervised regressor y_j = F^γ_k(j) (or its log) from the SAME 26-dim
observation, trained on trajectories generated by H2O and random (to cover feature
distributions under eviction), models: the v1 MLP and sklearn HistGradientBoosting; policy
= evict the m lowest predictions. It isolates whether RL's sequential credit assignment /
task term adds anything over "learn a good score, then top-k". If RL ≈ regressor in v1,
that is a legitimate, reportable v1 outcome; RL's raison d'être is v2+ (adaptive budgets,
hardware terms) — say so in the write-up rather than hide it.

Offline oracles (sim-only upper reference): (i) greedy future-mass — evict the m smallest
F_k(j); (ii) lookahead-H — smallest mass over the next H chunks (Belady flavour). Neither is
provably optimal for the additive-lost-mass objective (exact optimum is a min-cost flow
over eviction times; optional experiment); a learned policy beating both is a bug signal.

Success criterion for "RL earns its place" (v1): at B = 25 %, paired per-prompt bootstrap
CI vs H2O and vs the regressor, on sim lost mass AND real ΔNLL AND ≥ 2 task families;
RL must win outside the CI on real metrics, not only in sim.

## 7. Ablation plan (each: hypothesis → metric → decision rule)

1. Features: age-only → +attention (5–9) → +layer-max/disp (10–11) → +hits (12–14) →
   +norms (15–16) → +redundancy (17–18). Hypothesis: attention EMAs give most of the
   gain, layer-max helps needle. Metric: sim ℓ, needle retention. Drop features whose
   removal changes ℓ by < CI.
2. History depth: last-chunk only vs EMAs vs raw last-8-chunk vector (8 dims). Hypothesis:
   EMA fast+slow suffices.
3. Architecture: MLP vs DeepSets vs ISAB-2. Hypothesis: set-awareness matters more at low
   budgets (12.5 %).
4. Sampler/objective: per-slot vs sequence-level ratio; RB entropy vs H_1; R1 vs R2 reward;
   γ ∈ {0.98, 0.99, 0.995}; privileged critic on/off; warm start on/off.
5. Protection: S ∈ {0, 4, 8}, W ∈ {64, 128, 256, B/2}. Detects protected-token reliance (§8).
6. Global vs per-layer decisions (v3 preview, sim-only with per-group traces on ≤ 50
   prompts): does layer-uniform eviction leave a lot on the table? Quantify the oracle gap.
7. Sim-trained vs real-fine-tuned (late phase): PPO on Env B with R2 = −ΔNLL per chunk
   (needs a full-cache reference run; ~200 episodes at 4K–8K is affordable).
8. Generalisation: train on 4K–8K prompts, test 16K (real) and 32K (sim); train B = 25 %,
   test 12.5/37.5/50 %; needle positions unseen in training. Metric: real ΔNLL and task
   accuracy vs H2O at the same length.
9. Trace fidelity: reward/features from layer-mean only vs +layer-max; uint8-log vs fp16
   storage; sparsified (< 1e-4 zeroed) vs dense.

## 8. Failure modes and detectors (logged every update / every eval)

- Churn ("thrashing" analogue — eviction is irreversible, so the pathology is evicting
  tokens right after they become candidates or, inversely, hoarding old tokens): histogram
  of evict-age; fraction of evictions with age ≤ 2 chunks; retained-set age profile vs
  window/H2O. Alarm if > 80 % of evictions are the youngest candidates (policy = inverse
  window) or the oldest (= sliding window in disguise) while ℓ is not better than those.
- Protected-token reliance: eval with S = 0 → if ℓ collapses far more than for H2O, the
  policy never learned sink importance; keep S = 4 in deployment but report it.
- Degenerate/uniform scores: H̄ near log n_cand, score std → 0, Jaccard(policy, random)
  high, ℓ ≈ random. Alarm: H̄ > 0.9·max after 50 updates or std(s) < 0.05.
- Position overfitting: permutation importance (shuffle attention features across tokens
  at eval → if ℓ unchanged, the policy is positional); needle-position sweep; 8K→16K test.
- Reward hacking of the task term: general-text NLL and lost mass on unlabelled prompts
  must not degrade when λ_task > 0.
- Optimisation pathologies: approx-KL > 0.03/update, clip-frac > 0.3, value explained
  variance < 0.3 → lower lr / raise minibatch / check reward scale.
- Sim→real gap: E-proxy correlations and Jaccard of evict sets from sim vs real features
  below thresholds → fix features/reward before scaling.

## 9. PPO hyperparameters to start with

| hparam | value | why |
|--------|-------|-----|
| γ | 0.99 (0.995 for ≥ 16K) | an eviction's cost accrues over the rest of the episode; effective horizon 100 chunks ≈ 6.4K tokens covers 8K episodes; measure the future-mass half-life from traces (E-scale) and set γ so ≥ 90 % of a token's discounted future mass is inside the horizon |
| GAE λ | 0.95 | standard; rewards are dense under R1 |
| envs × steps | 32 × 128 = 4,096 decision-steps/update | ≈ 4,096 × 2044 × 26 × 2 B ≈ 440 MB fp16 obs buffer (fits beside no LLM in RAM); 8,192 if memory allows |
| minibatch / epochs | 256 decision-steps × 16 mb × 4 epochs | ~0.5 M token-rows per minibatch through a 20K-param MLP |
| lr | 3e-4 → linear to 3e-5, Adam(β2 = 0.999), eps 1e-5 | tiny nets, dense reward |
| clip ε | 0.2 (per slot); value clip 0.2 | per-slot ratios are well-scaled |
| entropy coef | 0.01 on H̄ (per-slot-normalised) | H̄ ≤ 7.6 nats → bonus ≤ 0.08 vs O(1) normalised policy loss; ablate {0, 0.003, 0.03} |
| value coef / grad clip | 0.5 / 0.5 | standard |
| advantage norm / reward scale | per-batch / fixed r_scale (§4.2) | bounded rewards; running-return norm as ablation |
| target-KL early stop | 0.02 (slot-averaged) | guards the 64-slot surrogate |
| training length | 2–5 M decision-steps (500–1,200 updates) | throughput estimate: rollout ~10–15 s, update ~60–90 s on CPU (≈ 4 TFLOP per update); MPS for the update ~10× faster — measure before committing (target ≥ 0.5 M decision-steps/h) |
| curriculum | 4K–8K prompts, B = 25 % → multi-budget {12.5, 25, 50 %} → 16K | budget-conditioned single policy after v1.0 |

Sanity gates before scaling (RL Engineer): beats random within 20 updates; matches H2O
when given only H2O's feature (attn_mean) and a top-m sampler; regressor warm start
reproduces regressor performance at update 0.

## 10. Upgrade path (sketch, concrete deltas)

- v2 budget-aware / adaptive occupancy: add a scalar sub-action per decision — target
  occupancy u ~ Beta(α(s), β(s)) rescaled to [u_min, 1]·B_max, m = max(0, n − uB_max);
  log π = log Beta + log PL(given m). Reward r = −quality − λ_mem·(|R_k|/N_full) −
  λ_lat·(attention-cost proxy ∝ |R_k|), λ_mem set by a Lagrangian dual on a per-episode
  average-occupancy target so "keep everything" is infeasible; B_max stays a hard rail;
  budget/target sampled per episode and given as input (universal policy). Real Pareto
  frontier then comes from the policy, not only from sweeping B.
- v3 per-layer(-group) policies: 4 layer groups (0–5, 6–11, 12–17, 18–23), shared network +
  4-dim group embedding, one PL sample per group with m_g; a softmax allocation head splits
  the global budget across groups (learned PyramidKV-like profiles). Requires per-group
  trace aggregates (4× trace size) and ragged per-layer cache lengths in Env B (batch = 1
  makes this feasible). Per-head is a study (retrieval heads), not a deployment target here.
- v4 chunk-hierarchical: level 1 PL over chunks (n/C ≈ 32 at B = 2K, ≈ 128 at 8K kept)
  choosing which chunks to open/evict wholesale; level 2 PL over tokens inside opened chunks;
  log π = sum of both levels (exact). Cuts candidates from ~2K to ~32 + 64·k_open; needed
  for 32K+ and for the transformer policy.
- Merging / compression: per-token categorical {KEEP, EVICT, MERGE→neighbour, Q4}; budget in
  bytes. Exact log-prob via two stages: independent Bernoulli compress/merge flags on kept
  tokens (exact), then PL eviction with m determined by the resulting byte count. Merging
  needs stored low-dim key projections (or real-model measurement) because merged-KV
  attention cannot be replayed from mass statistics — flag as a trace extension.

## 11. Open questions → resolving experiments

- E-sink: sink count S on Qwen2.5-0.5B (mass on positions 0..7 vs layer). Before Phase 2.
- E-scale: distribution of ℓ_k, F^γ, future-mass half-life, r_scale, λ_task calibration
  from ~50 traces under random/H2O. Before the first PPO run.
- E-proxy: §4.3 correlations. Gate for scaling RL.
- E-store: trace size vs fidelity (fp16 vs uint8-log vs sparse) at 8K/16K; expected 2 MB
  (mean+max, fp16) per 8K prompt, 8 MB per 16K — 500 prompts ≈ 1–4 GB → uint8-log/sparse
  likely required given ~9 GB free.
- E-throughput: sim + MLP decision-steps/hour on CPU vs MPS; decides rollout size and
  whether a fixed cheap candidate shortlist (e.g. exclude top-50 % by attn_ema_slow, still
  an exact PL over the shortlist) is needed at B ≥ 4096.

## 12. Recommended v1 spec

| item | choice |
|------|--------|
| decision | after every C = 64 tokens (prefill chunk / 64 decode steps); m = n − B, hard |
| protection | S = 4 sinks + current chunk (W = 64) hard-protected, count against B |
| state | tok 18 dims (§3.1) + glob 8 dims (§3.2); ϕ-transform + frozen obs-norm; identical FeatureState in sim and real |
| critic extras | 3 sim-only privileged globals (asymmetric critic; ablate) |
| action sampler | Plackett–Luce via Gumbel-top-k; per-slot exact logp (Z_U/T_j/D_j formula); per-slot clipped surrogate; RB entropy H̄ |
| reward | r_k = −Σ_{j∈E_k} F^γ_k(j)/r_scale (R1), + λ_task·gold-retained-fraction at T (λ_task = 1, ≤ 30 % labelled episodes); no memory/latency term (fixed by B) |
| policy net | MLP 26→128→128→1, 20,097 params, ~1–3 ms/decision CPU (≪ 20–40 ms decode step) |
| value net | token 26→128, mean⊕max pool ⊕ glob ⊕ priv → 128 → 1, ~38K params |
| PPO | γ 0.99, λ 0.95, 4,096 steps/update (32 envs × 128), mb 256, 4 epochs, lr 3e-4→3e-5, clip 0.2, ent 0.01, vf 0.5, grad-clip 0.5, target-KL 0.02 |
| eval policy | deterministic top-m; also stochastic |
| must beat | random, window, SnapKV-style, H2O, key-norm, supervised regressor (paired CIs, real ΔNLL + tasks) |
| upper refs | greedy future-mass oracle, lookahead-H oracle (sim) |
| gates | E-sink, E-scale, E-proxy (ρ ≥ 0.7), sanity gates §9 before any long run |
