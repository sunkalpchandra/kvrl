# DATA_SPEC — traces, features, datasets

## Trace (one per prompt) — `data/raw/<split>/<trace_id>.npz` + `data/raw/<split>/index.parquet`
Produced by `kvrl.collect` from a full-cache run with stats on (chunk C=64, dual attention).
| key | shape / dtype | meaning |
|-----|---------------|---------|
| token_ids | int32[T] | prompt + G generated tokens (T = P + G) |
| n_prompt, n_gen, chunk | scalars | P, G, C |
| attn_mean | float16[K, T] lower-triangular (keys ≤ chunk end) | A_k(j): attention mass from chunk k's queries on key j, mean over heads & layers, Σ_j = 1 |
| attn_lmax | float16[K, T] | layer-max of the head-mean mass |
| key_norm, value_norm | float16[T] | layer-mean L2 norms (computed at insertion) |
| adj_key_cos | float16[T] | layer-mean cosine(key_j, key_{j-1}) |
| gen_logprob | float16[G] | full-cache log-prob of each generated token (for ΔNLL) |
| critical_mask | bool[T] | task-critical tokens (from TaskInstance spans), all False for lm |
| meta (json string) | task, answers, seed, model, dtype, device, commit, timings |
Optional (layer study subset): `attn_group[4, K, T]` float16 for 4 layer groups.
Size: ≈ K·T·2·2 B ≈ 2 MB per 8K prompt (fp16 dense triangular, npz compressed ⇒ smaller).

## Processed / features
`FeatureState` computes obs on the fly from traces (no materialised feature store in v1);
`data/processed/regressor_<split>.parquet` holds sampled (obs, F^γ) rows for the supervised baseline.

## Evaluation data
Generated deterministically by `kvrl.eval.tasks` (seeded) — not stored; the eval config
(task, n, target_tokens, seed) fully identifies it. Gutenberg filler in `data/corpus/`.

## Splits
train: seeds 0–999 (tasks: lm 50%, needle/kv/multihop/dependency/code 10% each), context 2K–8K.
val: seeds 1000–1099. test: seeds 2000–2099 incl. unseen 16K length and unseen needle depths.
