#!/usr/bin/env python
"""E-proxy: does the simulator's lost-attention-mass predict real degradation?

For val traces × controllers × budgets we compute
  sim : lost_mass_decode from CacheSimEnv (trace replay)
  real: ΔNLL of the trace's own full-cache continuation under the evicted cache
        (teacher-forced), relative to the full-cache NLL measured in the same process,
        plus real lost mass is not observable cheaply, so ΔNLL is the target.
Reports Spearman ρ across all (trace, controller, budget) rows and per budget.
Rows are appended to runs/<id>/eproxy.jsonl; existing rows for a controller can be reused
(--reuse RUN_ID) so the RL policy can be added later on the same prompts.

    python scripts/eproxy.py --controllers window,random,h2o,snapkv,keynorm
    python scripts/eproxy.py --controllers rl --checkpoint checkpoints/ppo_mlp_v1.pt --reuse <run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvrl.controllers import make_controller
from kvrl.engine import InferenceEngine
from kvrl.models.hf_model import load_model
from kvrl.rl.train import load_traces
from kvrl.sim.env import CacheSimEnv, episode_budget, run_controller_episode
from kvrl.utils.tracker import start_run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/raw/val")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--controllers", default="window,random,h2o,snapkv,keynorm")
    ap.add_argument("--checkpoint", default="checkpoints/ppo_mlp_v1.pt")
    ap.add_argument("--budgets", default="0.125,0.25,0.5")
    ap.add_argument("--reuse", default=None, help="run id whose eproxy.jsonl rows to include")
    ap.add_argument("--max-forced", type=int, default=64)
    args = ap.parse_args()
    traces = load_traces(args.traces, limit=args.limit)
    ctrl_names = args.controllers.split(",")
    budgets = [float(b) for b in args.budgets.split(",")]
    model = load_model("qwen2.5-0.5b-instruct")
    engine = InferenceEngine(model, chunk_size=64, decide_every=64)
    from kvrl.controllers.learned import load_policy_checkpoint

    fcfg = None
    if "rl" in ctrl_names or "regressor" in ctrl_names:
        _, fcfg, _ = load_policy_checkpoint(args.checkpoint)
    env = CacheSimEnv(feature_cfg=fcfg) if fcfg else CacheSimEnv()
    rows = []
    if args.reuse:
        p = Path("runs") / args.reuse / "eproxy.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        print(f"[eproxy] reused {len(rows)} rows from {args.reuse}")
    cfg = {
        "traces": args.traces,
        "controllers": ctrl_names,
        "budgets": budgets,
        "limit": args.limit,
        "checkpoint": args.checkpoint,
        "reuse": args.reuse,
    }
    with start_run("eproxy", cfg, device=str(model.device)) as run:
        out = open(run.artifact_path("eproxy.jsonl"), "a")
        for r in rows:
            out.write(json.dumps(r) + "\n")
        full_nll: dict[str, float] = {r["trace_id"]: r["full_nll"] for r in rows}
        for tr in traces:
            ids = torch.from_numpy(tr.token_ids[: tr.n_prompt].astype(np.int64))
            forced = torch.from_numpy(
                tr.token_ids[tr.n_prompt : tr.n_prompt + args.max_forced].astype(np.int64)
            )
            if forced.numel() < 4:
                continue
            if tr.trace_id not in full_nll:
                res = engine.run(ids, make_controller("full"), budget=1 << 30, forced_ids=forced)
                full_nll[tr.trace_id] = res.nll
            done = {(r["trace_id"], r["controller"], r["budget_frac"]) for r in rows}
            for name in ctrl_names:
                for bf in budgets:
                    if (tr.trace_id, name, bf) in done:
                        continue
                    t0 = time.time()
                    ctrl = (
                        make_controller(name, checkpoint=args.checkpoint)
                        if name in ("rl", "regressor")
                        else make_controller(name)
                    )
                    sim = run_controller_episode(env, tr, ctrl, budget_frac=bf)
                    ctrl2 = (
                        make_controller(name, checkpoint=args.checkpoint)
                        if name in ("rl", "regressor")
                        else make_controller(name)
                    )
                    budget = episode_budget(bf, tr.n_prompt, tr.chunk)
                    real = engine.run(ids, ctrl2, budget=budget, forced_ids=forced)
                    row = {
                        "trace_id": tr.trace_id,
                        "task": tr.meta.get("task"),
                        "n_prompt": tr.n_prompt,
                        "controller": name,
                        "budget_frac": bf,
                        "budget": budget,
                        "sim_lost_decode": sim["lost_mass_decode"],
                        "sim_lost_mean": sim["lost_mass_mean"],
                        "sim_crit_retained": sim["crit_retained"],
                        "real_nll": real.nll,
                        "full_nll": full_nll[tr.trace_id],
                        "real_dnll": real.nll - full_nll[tr.trace_id],
                        "real_kv_peak_frac": real.kv_bytes_peak / max(1, real.kv_bytes_full),
                        "seconds": round(time.time() - t0, 1),
                    }
                    rows.append(row)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(
                        f"[eproxy] {tr.trace_id:22s} {name:8s} b={bf:<5} sim_lost={row['sim_lost_decode']:.4f} "
                        f"dNLL={row['real_dnll']:+.4f} ({row['seconds']}s)",
                        flush=True,
                    )
        out.close()
        df = pd.DataFrame(rows)
        df.to_parquet(run.artifact_path("eproxy.parquet"), index=False)
        summary = {"n": len(df)}
        rho, p = spearmanr(df.sim_lost_decode, df.real_dnll)
        summary["spearman_all"] = {"rho": float(rho), "p": float(p)}
        for bf, g in df.groupby("budget_frac"):
            r_, p_ = spearmanr(g.sim_lost_decode, g.real_dnll)
            summary[f"spearman_b{bf}"] = {"rho": float(r_), "p": float(p_), "n": len(g)}
        for c, g in df.groupby("controller"):
            r_, p_ = spearmanr(g.sim_lost_decode, g.real_dnll)
            summary[f"spearman_{c}"] = {"rho": float(r_), "p": float(p_), "n": len(g)}
        # ranking agreement: within each (trace, budget), rank controllers by sim vs real
        agree = []
        for (_, _), g in df.groupby(["trace_id", "budget_frac"]):
            if len(g) >= 3:
                r_, _ = spearmanr(g.sim_lost_decode, g.real_dnll)
                if not np.isnan(r_):
                    agree.append(r_)
        summary["within_prompt_rank_rho_mean"] = float(np.mean(agree)) if agree else None
        summary["by_controller_budget"] = (
            df.groupby(["controller", "budget_frac"])[["sim_lost_decode", "real_dnll"]]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        )
        run.finish(summary)
        print(
            json.dumps({k: v for k, v in summary.items() if k != "by_controller_budget"}, indent=1)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
