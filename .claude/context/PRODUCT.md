# PRODUCT  (dashboard / demo intent)

Aesthetic: minimal, dark, technical, whitespace; inspiration Linear / Vercel /
Anthropic research pages / GPU profilers. No cartoon UI, no gratuitous gradients.

Screens (planned):
1. Overview — model, context, budget; memory/latency/quality bars; tokens retained; policy.
2. Token retention view — per-token/chunk importance strip; click → token detail
   (age, attention received, predicted importance, action, confidence).
3. Decision trace — per step: state summary, action, confidence, reward, budget.
4. Experiment explorer — select model/dataset/context/budget/policy → quality,
   memory, latency, throughput vs baselines.
5. Pareto frontier — memory ↓ / latency ↓ / quality ↑ with policy toggles. Centrepiece.
6. Live demo — full-cache vs RL-cache side by side, streaming.

Backend: FastAPI serving `runs/` + live inference endpoint. Frontend: React+Vite+TS,
hash routing, static demo mode for GitHub Pages (JSON snapshots).


## Status (2026-08-16)
Implemented: FastAPI backend (`kvrl/server/app.py`: runs, pareto, bench, checkpoints, demo POST +
SSE stream, static mount) and React/Vite/TS/Tailwind frontend (`frontend/`): Live demo (controls,
side-by-side cards, token retention strip with click-for-detail, decision trace, run facts),
Pareto frontier (accuracy/fidelity/NLL vs KV%, series toggles, table), Latency & memory (bench
curves + hardware curve), Experiments (run list with provenance, filterable rows). Static
snapshot mode via `scripts/export_demo_snapshot.py` → `frontend/public/demo/*.json`.
