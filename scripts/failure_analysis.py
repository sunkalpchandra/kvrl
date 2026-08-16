#!/usr/bin/env python
"""Failure analysis on val traces (sim): critical-token evictions, evict-age histograms,
retention profiles, and RL feature permutation importance. Writes docs/figures/failure_*.png
and docs/failure_analysis.md.

    python scripts/failure_analysis.py --checkpoint checkpoints/ppo_mlp_v1.pt --budget 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvrl.controllers import make_controller
from kvrl.controllers.learned import OracleController, RLController, load_policy_checkpoint
from kvrl.eval.analysis import analyse_episode, permutation_importance
from kvrl.rl.train import load_traces
from kvrl.sim.env import CacheSimEnv

COLORS = {
    "rl": "#2ca02c",
    "h2o": "#1f77b4",
    "snapkv": "#e377c2",
    "window": "#ff7f0e",
    "random": "#bcbd22",
    "keynorm": "#9467bd",
    "oracle": "#333333",
    "regressor": "#17becf",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/raw/val")
    ap.add_argument("--checkpoint", default="checkpoints/ppo_mlp_v1.pt")
    ap.add_argument("--regressor", default="checkpoints/regressor_v1.pt")
    ap.add_argument("--budget", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default="docs")
    args = ap.parse_args()
    traces = load_traces(args.traces, limit=args.limit)
    if not traces:
        print("no traces")
        return 1
    policy, fcfg, ck = load_policy_checkpoint(args.checkpoint)
    env = CacheSimEnv(feature_cfg=fcfg, r_scale=ck.get("meta", {}).get("r_scale", 1.0))
    ctrls = {
        "rl": lambda: RLController(policy, fcfg),
        "h2o": lambda: make_controller("h2o"),
        "snapkv": lambda: make_controller("snapkv"),
        "window": lambda: make_controller("window"),
        "random": lambda: make_controller("random"),
        "keynorm": lambda: make_controller("keynorm"),
        "oracle": lambda: OracleController(env.future_mass),
    }
    if Path(args.regressor).exists():
        rp, rf, _ = load_policy_checkpoint(args.regressor)
        ctrls["regressor"] = lambda: RLController(rp, rf)
        ctrls["regressor"]().name = "regressor"
    rows = []
    for name, mk in ctrls.items():
        for tr in traces:
            c = mk()
            c.name = name
            rows.append(analyse_episode(env, tr, c, args.budget))
    df = pd.DataFrame(rows)
    figdir = Path(args.out) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Failure analysis (sim, val traces, budget {args.budget:.0%}) — auto-generated\n",
        f"{len(traces)} traces · checkpoint `{args.checkpoint}`\n",
        "| controller | lost mass (decode) | critical retained @question | critical retained (decode) | evict-age mean |",
        "|---|---|---|---|---|",
    ]
    summ = (
        df.groupby("controller")
        .agg(
            lost=("lost_mass_decode", "mean"),
            crit_q=("crit_retained_at_question", "mean"),
            crit_d=("crit_retained_decode", "mean"),
            age=("evict_age_mean", "mean"),
        )
        .sort_values("lost")
    )
    for c, r in summ.iterrows():
        lines.append(f"| {c} | {r.lost:.4f} | {r.crit_q:.3f} | {r.crit_d:.3f} | {r.age:.1f} |")
    # figure 1: retention profile
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for c, g in df.groupby("controller"):
        prof = np.nanmean(np.array(g.retention_profile.tolist()), axis=0)
        axes[0].plot(np.linspace(0, 1, len(prof)), prof, label=c, color=COLORS.get(c))
        hist = np.array(g.evict_age_hist.tolist()).sum(axis=0).astype(float)
        axes[1].plot(
            range(len(hist)),
            hist / max(1, hist.sum()),
            marker="o",
            ms=3,
            label=c,
            color=COLORS.get(c),
        )
    axes[0].set_xlabel("relative position in prompt")
    axes[0].set_ylabel("fraction retained at end")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].set_xticks(range(9))
    axes[1].set_xticklabels(
        ["0", "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127", "128+"], fontsize=7
    )
    axes[1].set_xlabel("age at eviction (decision steps)")
    axes[1].set_ylabel("fraction of evictions")
    fig.suptitle(f"Where do controllers keep tokens? (sim, budget {args.budget:.0%})", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "failure_retention_profile.png")
    plt.close(fig)
    # critical retention by task
    ct = (
        df[df.n_critical > 0]
        .groupby(["task", "controller"])["crit_retained_at_question"]
        .mean()
        .unstack()
    )
    if len(ct):
        lines.append("\n## Critical-token retention at question time, by task\n")
        lines.append(ct.to_markdown(floatfmt=".2f"))
        fig, ax = plt.subplots(figsize=(7, 3))
        ct.plot.bar(ax=ax, color=[COLORS.get(c) for c in ct.columns], width=0.85)
        ax.set_ylabel("critical tokens retained")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6, ncol=4, frameon=False)
        fig.tight_layout()
        fig.savefig(figdir / "failure_critical_by_task.png")
        plt.close(fig)
    # permutation importance
    pi = permutation_importance(
        env, traces[: min(8, len(traces))], RLController(policy, fcfg), args.budget
    )
    lines.append("\n## RL feature permutation importance (Δ lost mass when shuffled; sim)\n")
    lines.append("| feature | Δ lost mass |")
    lines.append("|---|---|")
    for r in pi:
        lines.append(f"| {r['feature']} | {r['delta']:+.4f} |")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([r["feature"] for r in pi][::-1], [r["delta"] for r in pi][::-1], color="#2ca02c")
    ax.set_xlabel("Δ lost attention mass (decode) when feature is shuffled")
    fig.tight_layout()
    fig.savefig(figdir / "failure_permutation_importance.png")
    plt.close(fig)
    (Path(args.out) / "failure_analysis.md").write_text("\n".join(lines) + "\n")
    (Path(args.out) / "failure_analysis.json").write_text(
        json.dumps(
            {"summary": summ.reset_index().to_dict(orient="records"), "permutation": pi}, indent=1
        )
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
