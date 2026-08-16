# DECISIONS  (append-only architectural decision log; newest at bottom)

Format: `D-NNN (date) — title` / Context / Decision / Consequences / Alternatives.

## D-001 (2026-08-16) — Project lives at ~/kvrl, package `kvrl`, repo sunkalpchandra/kvrl
Context: master build prompt names the CLI `python -m kvrl.*`.
Decision: package + repo both `kvrl`; frontend under `frontend/`.
Consequences: imports are `from kvrl...`; entrypoints `kvrl.train/evaluate/benchmark/demo`.

## D-002 (2026-08-16) — venv with `--system-site-packages`
Context: only ~9.5 GB free disk; torch 2.13 already in system Python 3.12; sibling
projects (gpu-optimizer) use the same pattern.
Decision: `.venv` reuses system torch/numpy/gymnasium; extra deps installed inside.
Consequences: `requirements.txt` still pins everything for clean machines; setup.sh
supports both modes.

## D-003 (2026-08-16) — Simulator-first RL; real inference for validation
Context: 8 GB laptop; running the LLM inside the RL loop is infeasible for millions
of transitions.
Decision: Env A (trace-replay simulator) trains; Env B (real HF inference with real
cache manipulation) validates and (late phase) fine-tunes.
