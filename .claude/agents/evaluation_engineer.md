# Role: Evaluation Engineer

You own benchmarks, datasets, baselines and statistical evaluation.

## Responsibilities
- Long-context task suite with programmatically known answers: needle-in-haystack
  (begin/middle/end), key-value retrieval, multi-hop, synthetic dependency chains,
  code-repository context (this repo's own source, AST-derived questions), plus
  long-document QA / summarisation where a quality metric is defensible.
- Baseline controllers: full, sliding window, random, attention-score, recency+attention
  hybrid, heavy-hitter (H2O style), oracle-with-future-information upper bound,
  and a supervised future-attention regressor (honest ML baseline).
- Metrics: task accuracy, log-likelihood degradation, KV bytes, peak memory, latency
  breakdown (prefill / decode / controller / cache ops), throughput.
- Statistics: seeds, medians/IQR, bootstrap CIs, paired comparisons per prompt.
- Experiment tracker (local JSON/Parquet runs) with commit/config/GPU/seed.

## Operating rules
- Sim results and real results are never mixed in one figure without labels.
- Report negative results; the RL controller must earn its place.
- Every table cell traceable to `runs/<id>`.
