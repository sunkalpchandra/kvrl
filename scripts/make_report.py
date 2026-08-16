#!/usr/bin/env python
"""Generate figures + markdown tables from tracked runs (README results come from here).

    python scripts/make_report.py [--eval RUN_ID] [--bench RUN_ID] [--train RUN_ID]

Outputs docs/figures/*.png and docs/results.md. Nothing is typed by hand: every number is read
from runs/<id>/ (results.parquet, bench.parquet, metrics.jsonl).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvrl.utils.tracker import list_runs, load_run

FIG = Path("docs/figures")
COLORS = {
    "full": "#7f7f7f",
    "rl": "#2ca02c",
    "h2o": "#1f77b4",
    "snapkv": "#e377c2",
    "window": "#ff7f0e",
    "random": "#bcbd22",
    "keynorm": "#9467bd",
    "regressor": "#17becf",
    "tova": "#8c564b",
    "hybrid": "#7fbf7f",
}
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def latest(kind: str, run_id: str | None):
    runs = [r for r in list_runs(kind=kind) if r.get("meta", {}).get("status") == "finished"]
    if run_id:
        runs = [r for r in runs if r["meta"]["run_id"] == run_id]
    return runs[-1] if runs else None


def pareto_figures(run) -> list[str]:
    df = pd.read_parquet(Path(run["dir"]) / "results.parquet")
    lines = [
        f"### Real-model evaluation — run `{run['meta']['run_id']}` (commit `{(run['meta'].get('commit') or '')[:8]}`, "
        f"{run['meta'].get('device_info', {}).get('gpu', '')}, {len(df)} rows)\n"
    ]
    g = df.groupby(["controller", "budget_frac"])
    tab = []
    for (c, b), gg in g:
        acc = gg["correct"].dropna().astype(float)
        tab.append(
            {
                "controller": c,
                "budget": "100%" if c == "full" else f"{b:.0%}",
                "n": len(gg),
                "accuracy": acc.mean() if len(acc) else float("nan"),
                "nll": gg["nll"].mean(),
                "fidelity": gg["fidelity"].mean(),
                "kv_peak": gg["kv_peak_frac"].mean(),
                "model_s": gg["model_s"].median(),
                "ctrl_s": gg["controller_s"].median(),
                "compact_s": gg["compact_s"].median(),
            }
        )
    t = pd.DataFrame(tab).sort_values(["budget", "controller"], ascending=[False, True])
    lines.append(
        "| controller | budget | n | accuracy | NLL | fidelity | KV peak | model s | ctrl s | compact s |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in t.iterrows():
        lines.append(
            f"| {r.controller} | {r.budget} | {r.n} | {r.accuracy:.3f} | {r.nll:.3f} | {r.fidelity:.3f} | "
            f"{r.kv_peak:.0%} | {r.model_s:.2f} | {r.ctrl_s:.3f} | {r.compact_s:.3f} |"
        )
    # figure: accuracy / nll / fidelity vs kv%, and accuracy vs MEASURED total latency
    med_lat = df.groupby(["controller", "budget_frac"])["total_s"].median()
    t["total_s"] = [
        med_lat.get((r.controller, 1.0 if r.controller == "full" else float(r.budget.rstrip("%")) / 100), float("nan"))
        for _, r in t.iterrows()
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for c, gg in t.groupby("controller"):
        gg = gg.sort_values("kv_peak")
        kw = dict(color=COLORS.get(c, None), marker="o", label=c, ms=4)
        axes[0].plot(gg.kv_peak, gg.accuracy, **kw)
        axes[1].plot(gg.kv_peak, gg.nll, **kw)
        axes[2].plot(gg.kv_peak, gg.fidelity, **kw)
        axes[3].plot(gg.total_s, gg.accuracy, **kw)
    for ax, yl in zip(
        axes[:3], ["task accuracy", "NLL (natural continuation)", "fidelity vs full (ROUGE-L)"]
    ):
        ax.set_xlabel("peak KV memory (fraction of full)")
        ax.set_ylabel(yl)
        ax.set_xlim(0, 1.05)
    axes[3].set_xlabel("measured total latency / prompt (s, median incl. controller + cache ops)")
    axes[3].set_ylabel("task accuracy")
    axes[0].set_ylim(0, 1.02)
    axes[2].set_ylim(0, 1.02)
    axes[3].set_ylim(0, 1.02)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Quality vs KV memory and vs measured latency (real model, all tasks pooled)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "pareto.png")
    plt.close(fig)
    # per-task accuracy heat-ish table
    bt = (
        df[df.correct.notna()]
        .groupby(["task", "controller", "budget_frac"])["correct"]
        .mean()
        .reset_index()
    )
    if len(bt):
        lines.append("\n#### Accuracy by task (25% budget)\n")
        sub = bt[(bt.budget_frac == 0.25) | (bt.controller == "full")]
        piv = sub.pivot_table(index="task", columns="controller", values="correct")
        lines.append(piv.to_markdown(floatfmt=".2f"))
        fig, ax = plt.subplots(figsize=(7, 3))
        piv.plot.bar(ax=ax, color=[COLORS.get(c) for c in piv.columns], width=0.8)
        ax.set_ylabel("accuracy @ 25% budget")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, ncol=4, frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / "accuracy_by_task.png")
        plt.close(fig)
    # paired comparisons
    paired = (run.get("results") or {}).get("paired", [])
    if paired:
        lines.append("\n#### Paired comparisons (per prompt, bootstrap 95% CI)\n")
        lines.append(
            "| controller | budget | vs | metric | mean diff | CI | win rate | n | significant |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for p in paired:
            lines.append(
                f"| {p['controller']} | {p['budget_frac']:.0%} | {p['vs']} | {p['metric']} | {p['mean_diff']:+.4f} | "
                f"[{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}] | {p['win_rate']:.2f} | {p['n']} | {'yes' if p['significant'] else 'no'} |"
            )
    return lines


def bench_figures(run) -> list[str]:
    df = pd.read_parquet(Path(run["dir"]) / "bench.parquet")
    df = df[df.ok]
    lines = [
        f"\n### Latency / memory — run `{run['meta']['run_id']}` ({run['meta'].get('device_info', {}).get('gpu', '')}, "
        f"medians of {int(df.repeats.max()) if len(df) else 0} repeats)\n"
    ]
    lines.append(
        "| context | controller | budget | prefill s | decode ms/tok | controller s | compact s | KV peak | peak alloc MB |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in df.sort_values(["context", "controller"]).iterrows():
        lines.append(
            f"| {r.context} | {r.controller} | {r.budget_frac:.0%} | {r.prefill_s_median:.2f} | {r.decode_ms_per_tok_median:.1f} | "
            f"{r.controller_s_median:.3f} | {r.compact_s_median:.3f} | {r.kv_peak_frac:.0%} | {r.peak_allocated_mb_median:.0f} |"
        )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for c, gg in df.groupby("controller"):
        gg = gg.sort_values("context")
        kw = dict(color=COLORS.get(c), marker="o", label=c, ms=4)
        axes[0].plot(gg.context, gg.prefill_s_median, **kw)
        axes[1].plot(gg.context, gg.decode_ms_per_tok_median, **kw)
        axes[2].plot(gg.context, gg.kv_bytes_peak / 2**20, **kw)
    for ax, yl in zip(axes, ["prefill (s)", "decode (ms / token)", "peak KV bytes (MB)"]):
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel(yl)
        ax.set_xscale("log", base=2)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Latency and memory vs context length (real hardware)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "latency_memory.png")
    plt.close(fig)
    curve = (run.get("results") or {}).get("decode_curve")
    if curve:
        c = pd.DataFrame(curve)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(c.cache_len, c.decode_ms_per_tok_median, marker="o", color="#333")
        ax.fill_between(c.cache_len, c.p25, c.p75, alpha=0.2, color="#333")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("KV cache length (tokens)")
        ax.set_ylabel("decode ms / token (full cache)")
        ax.set_title("Hardware curve: decode cost vs cache length", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / "decode_curve.png")
        plt.close(fig)
        try:
            from kvrl.bench.cost_model import fit_decode_cost

            cm = fit_decode_cost(curve, device=str(run["meta"].get("device_info", {}).get("gpu", "")))
            lines.append(f"\nFitted decode cost model: {cm.ms_per_token_base:.2f} ms/token + {cm.ms_per_token_per_1k:.3f} ms per 1K cached tokens (R² = {cm.r2:.3f}, {cm.n_points} points).\n")
        except Exception as e:
            lines.append(f"\n_cost model fit failed: {e!r}_\n")
        lines.append("\n#### Decode cost vs cache length (full cache)\n")
        lines.append("| cache length | decode ms/tok (median) | IQR |")
        lines.append("|---|---|---|")
        for _, r in c.iterrows():
            lines.append(
                f"| {int(r.cache_len)} | {r.decode_ms_per_tok_median:.1f} | {r.p25:.1f}–{r.p75:.1f} |"
            )
    return lines


def train_figures(run) -> list[str]:
    ms = [m for m in run.get("metrics", []) if "ep_return_mean" in m]
    ev = [m for m in run.get("metrics", []) if "eval" in m]
    if not ms:
        return []
    m = pd.DataFrame(ms)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].plot(m.steps, m.ep_return_mean, color="#2ca02c")
    axes[0].set_ylabel("episode return (sim)")
    axes[1].plot(m.steps, m.ep_lost_mass_decode, color="#2ca02c", label="rl (stochastic rollout)")
    axes[1].set_ylabel("lost attention mass, decode (sim)")
    axes[2].plot(m.steps, m.approx_kl, color="#555")
    axes[2].set_ylabel("approx KL / update")
    for ax in axes:
        ax.set_xlabel("decision steps")
    if ev:
        rows = []
        for e in ev:
            for r in e["eval"]:
                rows.append({"steps": e["step"], **r})
        e = pd.DataFrame(rows)
        for c, gg in e[e.budget_frac == 0.25].groupby("controller"):
            axes[1].plot(
                gg.steps,
                gg.lost_mass_decode,
                marker="o",
                ms=3,
                ls="--",
                label=f"{c} (val, det.)",
                color=COLORS.get(c),
            )
        axes[1].legend(fontsize=6, frameon=False)
    fig.suptitle(f"PPO training in the simulator — run {run['meta']['run_id']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "training.png")
    plt.close(fig)
    res = run.get("results", {})
    lines = [
        f"\n### PPO training — run `{run['meta']['run_id']}`\n",
        f"- updates {res.get('updates')}, decision steps {res.get('steps')}, episodes {res.get('episodes')}, "
        f"train time {res.get('train_seconds')} s, policy params {res.get('policy_params')}",
        f"- best val lost-mass (decode, sim): {res.get('best_val_lost_mass_decode')}",
    ]
    best = Path(run["dir"]) / "best_eval.json"
    if best.exists():
        b = pd.DataFrame(json.loads(best.read_text()))
        piv = b.pivot_table(index="controller", columns="budget_frac", values="lost_mass_decode")
        lines.append("\nSim lost-mass (decode) on val traces, deterministic policies:\n")
        lines.append(piv.to_markdown(floatfmt=".4f"))
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=None)
    ap.add_argument("--bench", default=None)
    ap.add_argument("--train", default=None)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    out = ["# Results (auto-generated by scripts/make_report.py — do not edit)\n"]
    for kind, fn, rid in (
        ("eval", pareto_figures, args.eval),
        ("bench", bench_figures, args.bench),
        ("train", train_figures, args.train),
    ):
        run = latest(kind, rid)
        if run is None:
            out.append(f"\n_no finished {kind} run found_\n")
            continue
        run = load_run(run["dir"])
        try:
            out.extend(fn(run))
        except Exception as e:  # keep going; report the failure honestly
            out.append(f"\n_{kind} report failed: {e!r}_\n")
    Path("docs/results.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
