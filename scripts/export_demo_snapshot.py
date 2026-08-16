#!/usr/bin/env python
"""Export dashboard JSON snapshots to frontend/public/demo/ for static hosting (GitHub Pages).

    python scripts/export_demo_snapshot.py

Writes runs.json, pareto.json, bench.json, checkpoints.json, demo.json (if a demo was run) and
run_<id>.json / rows_<id>.json for eval + bench runs. Every file is a verbatim copy of what the
API serves — nothing is edited by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from kvrl.server.app import app


def main() -> int:
    out = Path("frontend/public/demo")
    out.mkdir(parents=True, exist_ok=True)
    c = TestClient(app)

    def dump(name: str, path: str) -> None:
        r = c.get(path)
        if r.status_code == 200:
            (out / name).write_text(json.dumps(r.json(), default=str))
            print(f"[export] {name} ({len(r.content) / 1024:.0f} KB)")
        else:
            print(f"[export] skip {name}: {r.status_code}")

    dump("runs.json", "/api/runs")
    dump("pareto.json", "/api/pareto")
    dump("bench.json", "/api/bench")
    dump("checkpoints.json", "/api/checkpoints")
    dump("demo.json", "/api/demo/snapshot")
    for run in c.get("/api/runs").json():
        if run["kind"] in ("eval", "bench", "train"):
            rid = run["run_id"]
            dump(f"run_{rid}.json", f"/api/runs/{rid}")
            if run["kind"] != "train":
                dump(f"rows_{rid}.json", f"/api/runs/{rid}/rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
