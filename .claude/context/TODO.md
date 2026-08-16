# TODO (prioritised)

## Finishing this session
- [ ] long-context eval (8K/16K) → report section; record E-010
- [ ] re-benchmark with the faster decode loop (perf commit) → regenerate report
- [ ] export dashboard snapshot (scripts/export_demo_snapshot.py) after a demo run
- [ ] final STATUS/EXPERIMENTS; README numbers re-injected

## Next (v1.x)
- [ ] ablations with the warm-start protocol or ≥40K steps (E-009 shows 8K from scratch is too weak)
- [ ] reward that rewards keeping *not-yet-attended* critical tokens (needle failure mode): e.g.
      key-norm prior in the reward, or a retrieval-head layer-group term
- [ ] E-proxy rows for rl/regressor (scripts/eproxy.py --controllers rl --reuse <run>)
- [ ] generalisation table: train ≤4K → 8K real (partially in failure analysis; add real numbers)

## Later (v2+)
- [ ] adaptive occupancy (Beta sub-action + Lagrangian memory term) → policy-driven Pareto
- [ ] per-layer-group policies (ragged caches supported by the engine already)
- [ ] hierarchical chunk→token decisions for 32K+
- [ ] merge/compress actions (byte budget)
- [ ] batch > 1 (padding masks with the custom attention)
- [ ] CUDA validation run
