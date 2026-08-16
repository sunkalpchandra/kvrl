# TODO (prioritised)

## Now (Phase 4/5)
- [ ] finish trace collection; re-run `kvrl.collect` to fill missing code traces (packing fix)
- [ ] full PPO run (configs/train.yaml, 60K steps) → checkpoints/ppo_mlp_v1.pt; record in EXPERIMENTS.md
- [ ] E-proxy: sim lost-mass vs real ΔNLL / accuracy correlation across (prompt, policy, budget)
- [ ] real evaluation matrix (configs/evaluate.yaml) incl. RL; paired CIs vs h2o
- [ ] benchmark run (configs/benchmark.yaml): latency/memory vs context, decode curve
- [ ] regressor baseline training (scripts/train_regressor.py) — honest ML baseline
- [ ] slow integration test on the real model (pytest -m slow)

## Next (Phases 6–12)
- [ ] ablations script (features/history/arch/protection/global-vs-layer) with plots
- [ ] generalisation: train 2K–4K → test 8K; unseen depths
- [ ] failure analysis tools (critical-token evictions, evict-age histograms)
- [ ] hardware-aware objective (v2: occupancy sub-action + latency term)
- [ ] dashboard: FastAPI + React (overview, retention strip, decision trace, explorer, Pareto)
- [ ] perf: profile controller/feature/compaction; batch policy decisions
- [ ] README with real numbers + diagrams; setup.sh dry-run; Dockerfile
