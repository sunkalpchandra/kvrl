# kvrl — RL-managed KV cache for long-context Transformer inference

This file is read at the start of every session. It is the entry point to the
project's persistent engineering context. **Do not rely on conversation history
for architectural decisions — they live in `.claude/context/`.**

## Session start checklist

1. Read `.claude/context/PROJECT.md` (goal, scope, non-goals)
2. Read `.claude/context/ARCHITECTURE.md` (system design; authoritative)
3. Read `.claude/context/STATUS.md` (what works right now, what is broken)
4. Read `.claude/context/TODO.md` (prioritised next steps)
5. Before changing architecture: read `DECISIONS.md` (append, never rewrite history)
6. Before running or designing experiments: read `EXPERIMENTS.md`, `BENCHMARKS.md`
7. After significant work: update `STATUS.md`, `TODO.md`; log decisions/bugs/experiments

## Hard rules

- **No fabricated numbers.** Every number in docs/README/dashboard must come from a
  run whose config + commit + timestamp are recorded in `EXPERIMENTS.md` or in
  `runs/`. If a result is bad, report it and investigate.
- **Simulated ≠ measured.** Anything from the cache simulator is labelled `sim`;
  anything from real inference is labelled `real`. Never mix them in one bar.
- **Correctness first.** Real-cache eviction is verified against a masked
  full-cache reference; budget=100% must reproduce standard HF greedy output.
- **Small commits, pushed often.** Conventional prefixes (feat/fix/test/docs/exp/build).
  Run `make test` (or `pytest -q -m "not slow"`) before each commit.
- **Never commit** model weights, traces, datasets, secrets, node_modules.
- **No dead code**: no `TODO: implement later`, no `pass` stubs, no unused imports
  (`ruff check .` must be clean).

## Environment (verified 2026-08-16)

- macOS 14.6.1, Apple M2, 8 GB unified memory, ~9.5 GB free disk (tight!)
- No CUDA. `torch.backends.mps` available (recommended max ≈ 5.33 GiB). CPU fallback.
- Python 3.12.0 (framework build), venv at `.venv` (`--system-site-packages`,
  reusing system torch 2.13.0, numpy 2.2.5, gymnasium 1.3.0, pandas, scipy,
  matplotlib, pytest, sklearn). Extra deps installed into the venv.
- transformers 5.x (v5 Cache API — `DynamicCache.layers[i].keys/values`; verify
  at runtime, see `kvrl/models/hf_model.py`).
- Node 22.14 / npm 10.9 for the dashboard.
- GitHub: `sunkalpchandra/kvrl` (gh CLI authenticated).

## Layout

```
kvrl/            Python package (models, cache, controllers, sim, rl, eval, bench, cli)
configs/         YAML experiment configs
tests/           pytest (unit, integration, regression); `-m slow` for model runs
scripts/         one-off / reproducible pipelines (trace collection, ablations)
frontend/        React + Vite + TS dashboard
data/            gitignored: raw traces, processed, features, eval
runs/            gitignored: experiment tracker output (json + parquet)
checkpoints/     trained policies (small .pt files may be committed if < 5 MB)
docs/            diagrams, design notes
.claude/         this context system + agent role prompts + workflows
```

## Roles

Specialised subagent prompts live in `.claude/agents/`. Use them for design
reviews before major phases; the lead synthesises and records the outcome in
`DECISIONS.md` before implementing.
