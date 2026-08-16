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

## Measurement notes (2026-08-16)
- Machine under memory pressure (Chrome/VS Code; swap 4–7 GB): single-run latencies vary ±50%.
  Only medians of ≥3 repeats after 1 warmup are quoted; IQRs are stored alongside.
- Controller overhead after moving metadata to CPU and lazy norms: ~0.2–0.3 ms per decision for
  heuristics (from 1–30 ms on MPS tensors); compaction 3–5 ms typical, ~100 ms first call.
- Attention statistics ("dual" path) ≈ 3× the fused attention at 2K (isolated micro-bench);
  in the engine ≈ 1.6–2× end-to-end prefill (h2o 298 vs full 615 tok/s at 2K, one run).
