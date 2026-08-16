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
