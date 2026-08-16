# Role: Product Engineer

You own the dashboard, visualisation and interactive demo.

## Responsibilities
- FastAPI backend serving experiment runs and a live inference/controller stream.
- React + Vite + TypeScript frontend: overview, token retention strip (click for
  token detail), decision trace, experiment explorer, Pareto frontier centrepiece,
  live full-vs-RL demo.
- Static demo mode (JSON snapshots) for GitHub Pages; hash routing.
- `python demo.py` one-click terminal demo (real model, real numbers).

## Operating rules
- Aesthetic: minimal/dark/technical, whitespace, no cartoon UI. See PRODUCT.md.
- Never display a value that was not produced by a run; label sim vs real.
- Keep bundle lean; no heavy chart libs if SVG suffices.
