# TODO (prioritised)

## Finishing this session
- [x] long-context eval (E-010), benchmark v4, README results re-injected
- [ ] ablations (warm-start protocol) → docs/ablations.md
- [ ] dashboard snapshot re-export after the final report

## Next (v1.x)
- [ ] ablations with the warm-start protocol or ≥40K steps (E-009 shows 8K from scratch is too weak)
- [ ] real-inference fine-tuning (Env B) with task accuracy as the reward — the sim objective
      (attention mass + token penalties) demonstrably is not task accuracy (E-014/E-017)
- [ ] E-proxy rows for rl/regressor (scripts/eproxy.py --controllers rl --reuse <run>)
- [ ] generalisation table: train ≤4K → 8K real (partially in failure analysis; add real numbers)

## Later (v2+)
- [ ] adaptive occupancy (Beta sub-action + Lagrangian memory term) → policy-driven Pareto
- [ ] per-layer-group policies (ragged caches supported by the engine already)
- [ ] hierarchical chunk→token decisions for 32K+
- [ ] merge/compress actions (byte budget)
- [ ] batch > 1 (padding masks with the custom attention)
- [ ] CUDA validation run
