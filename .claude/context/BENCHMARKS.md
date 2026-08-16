# BENCHMARKS  (how we measure; canonical numbers with provenance)

## Measurement protocol (to be finalised in Phase 1)
- Device sync before/after timing (`torch.mps.synchronize()` / `torch.cuda.synchronize()`).
- Warmup iterations discarded; report median + IQR over N iterations.
- Memory: allocated + peak (`torch.mps.current_allocated_memory()`,
  `torch.mps.driver_allocated_memory()`; CUDA equivalents), plus analytic KV bytes.
- Total latency = model forward + controller inference + cache manipulation; each
  reported separately AND summed.

## Canonical tables
(none yet)
