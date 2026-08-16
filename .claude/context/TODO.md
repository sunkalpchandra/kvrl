# TODO (prioritised; keep short, move done items to STATUS/EXPERIMENTS)

## Phase 0 — bootstrap
- [x] inspect environment
- [x] create .claude context system
- [ ] agent role prompts
- [ ] ML Architect + Inference Engineer independent designs → reconcile → ARCHITECTURE.md
- [ ] pick model (candidate: Qwen/Qwen2.5-0.5B-Instruct — 32K ctx, GQA, ~1 GB bf16)
- [ ] GitHub repo + first push

## Phase 1 — model + full-cache baseline + benchmark harness
- [ ] model registry / HF wrapper with custom attention (stats capture, chunked prefill)
- [ ] manual decode loop with explicit position handling
- [ ] cache abstraction: inspect / compact (index_select) per layer
- [ ] correctness test: budget=100% == HF greedy
- [ ] benchmark harness (latency, memory; MPS + CUDA + CPU)

## Phase 2 — traces + simulator
## Phase 3 — heuristic baselines
## Phase 4 — PPO
## Phase 5 — real integration
## Later — representation, hierarchy, hardware-aware, suite, dashboard, perf, docs
