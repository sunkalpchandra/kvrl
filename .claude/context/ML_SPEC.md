# ML_SPEC — RL formulation (v1), distilled from design_ml_architect.md (full rationale there)

## MDP
- Step: after each chunk of C=64 tokens enters the cache (prefill chunk / 64 decode steps).
  Cache holds n = |R_{k-1}| + C; policy evicts m = max(0, n − B). Peak occupancy B + C.
- Episode: prompt + full-cache greedy continuation (teacher-forced from trace, G ≤ 256).
- Protected (outside action set, count against B): S=4 sink slots, current chunk (W=64).

## Action: Plackett–Luce set sampling (Gumbel-top-k)
- Scores s_i per candidate; g_i = s_i + Gumbel; evict list = top-m of g (ordered).
- Per-slot log-prob: p_j = s_{i_j}; T_j = reverse-logcumsumexp of picked scores; Z_U =
  logsumexp over unpicked; D_j = logaddexp(Z_U, T_j); logp_j = p_j − D_j.
- PPO surrogate over slots with shared advantage: L = −(1/m)Σ_j min(ρ_j Â, clip(ρ_j) Â).
- Entropy: Rao–Blackwellised per-slot estimator H̄ (fallback H_1). Eval: deterministic top-m.
- Tests: Σπ=1 exhaustively (n=5,m=2); Gumbel-top-k frequencies vs PL probs; masking; grads.

## Observation (identical code path in sim and real: `kvrl/features/FeatureState`)
Per token (18): age_log, rel_pos, pos_log, is_generated, attn_last, attn_ema_fast(0.5),
attn_ema_slow(0.9), attn_mean, attn_max, attn_lastmax_layer, attn_disp, hit_rate,
since_hit, chunk_share, key_norm(z), value_norm(z), adj_key_cos, nbr_evicted.
Attention values pass through ϕ(a; n) = log1p(a·n)/log1p(32768) (length-invariant).
Global (8): budget_frac, occupancy, evict_frac, ctx_log, step_frac, phase, remaining_gen,
chunk_entropy. Critic-only privileged (sim): committed future loss, retained future mass,
gold-retained fraction.

## Reward (sim, v1 quality-only; budget hard ⇒ memory fixed)
- r_k = −(1/r_scale) Σ_{j∈E_k} F^γ_k(j), F^γ_k(j) = Σ_{k'>k} γ^{k'−k} A^{max}_{k'}(j)  (R1)
  where A^max is the LAYER-MAX head-mean attention mass (D-009: E-proxy ρ=0.62 vs 0.44 for
  the layer-mean variant). The layer-mean lost mass is still logged for comparison.
- Telescoping identity with delayed lost-mass reward R2 (r_k = −ℓ_{k+1}/r_scale) — unit test.
- Terminal: + λ_task · (fraction of gold-span tokens retained through the answer), λ_task=1,
  labelled episodes ≤ 30% of batch. r_scale = mean|r_k| under random policy at B=25%.
- Validation on real model (E-proxy): Spearman ρ between sim lost mass and real ΔNLL / task
  accuracy across (prompt, policy, budget); gate ρ ≥ 0.7 before scaling RL.

## Networks
- Policy v1: MLP [tok18 ⊕ glob8]=26 → 128 → 128 → 1 (SiLU), 20,097 params; warm-started from
  the supervised future-mass regressor (D-009; from-scratch run E-003 stayed near-uniform).
- Value v1: token 26→128 SiLU, mean⊕max pool ⊕ glob ⊕ priv → 128 → 1 (~38K params).
- v1.5 DeepSets; v2 ISAB set-transformer (2 blocks, 32 inducing pts, d=64, ~75K params).

## PPO defaults
γ 0.99 (0.995 ≥16K), λ 0.95, 4096 decision-steps/update (32 envs × 128), minibatch 256,
4 epochs, lr 3e-4 → 3e-5, clip 0.2 (per slot), value clip 0.2, entropy 0.01 (H̄), vf 0.5,
grad-clip 0.5, target-KL 0.02, advantage normalisation per batch. Sanity gates: beats
random within 20 updates; matches H2O when given only attn_mean; regressor warm-start
reproduces regressor at update 0.

## Baselines & upper references
random · window (S sinks + recent) · snapkv (last-chunk attention) · h2o (cum. attention,
half recent) · tova (min last-query attention) · keynorm · hybrid (recency+attention) ·
regressor (supervised F^γ predictor on same 26 dims, top-k) · oracle (greedy future mass,
sim) · lookahead-H oracle. Success: RL beats H2O and regressor outside paired CI on real
ΔNLL and ≥2 task families at B=25%. If not, report it.

## Ablations (see design doc §7): features / history / architecture / sampler & reward /
protection S,W / global vs per-layer / sim vs real fine-tune / generalisation (4K–8K → 16K,
budgets, needle depths) / trace fidelity.

## Failure detectors: evict-age histogram (inverse-window / window-in-disguise), S=0
stress test, uniform-score alarm (H̄ > 0.9 max, std<0.05), permutation importance,
task-term reward hacking (unlabelled NLL), optimisation pathologies (KL, clip-frac, EV).

## Upgrade path: v2 Beta occupancy sub-action + Lagrangian memory term; v3 4 layer groups
with allocation head (ragged caches); v4 two-level PL (chunks → tokens); merge/compress
as categorical per-token with byte budget (needs stored key projections).
