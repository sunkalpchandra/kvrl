#!/usr/bin/env python
"""Merge several evaluation runs into one (for reporting): rows are keyed by
(task, seed, target_tokens, controller, budget_frac); later runs win on duplicates.
Prompts are deterministic given (task, tokens, seed, n), so a later run that adds a controller
on the same jobs is directly comparable per prompt.

    python scripts/merge_eval_runs.py --runs RUN_A RUN_B --out RUN_ID
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvrl.eval.runner import summarize_results
from kvrl.utils.tracker import git_info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="new run id (directory under runs/)")
    args = ap.parse_args()
    frames, metas = [], []
    for rid in args.runs:
        d = Path("runs") / rid
        df = pd.read_parquet(d / "results.parquet")
        df["source_run"] = rid
        frames.append(df)
        metas.append(json.loads((d / "meta.json").read_text()))
    df = pd.concat(frames, ignore_index=True)
    key = ["task", "seed", "target_tokens", "controller", "budget_frac"]
    before = len(df)
    df = df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    # keep only prompts present in every source (paired comparisons need identical prompt sets)
    out = Path("runs") / args.out
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "results.parquet", index=False)
    summary = summarize_results(df)
    summary["merged_from"] = args.runs
    summary["rows_before_dedup"] = before
    (out / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    meta = {"run_id": args.out, "kind": "eval", "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "finished", "merged_from": args.runs, "device_info": metas[0].get("device_info"),
            "seed": metas[0].get("seed"), **git_info()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (out / "config.yaml").write_text("merged_from:\n" + "".join(f"  - {r}\n" for r in args.runs))
    print(f"merged {before} rows -> {len(df)} into runs/{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
