#!/usr/bin/env python
"""Automated ablations: short PPO trainings with config deltas, compared on val traces (sim).

    python scripts/ablate.py --steps 20000 [--only features,arch]

Each ablation is a normal tracked training run (runs/<id>) with the delta recorded in the
config; results are collected into docs/ablations.md + docs/figures/ablations.png.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvrl.features import TOKEN_FEATURES
from kvrl.rl.train import train
from kvrl.utils.config import load_config
from kvrl.utils.tracker import start_run

ATTN = [
    "attn_last",
    "attn_ema_fast",
    "attn_ema_slow",
    "attn_mean",
    "attn_max",
    "attn_lastmax_layer",
    "attn_disp",
]
HIST = ["attn_ema_fast", "attn_ema_slow", "attn_mean", "attn_max", "hit_rate", "since_hit"]
NORMS = ["key_norm", "value_norm", "adj_key_cos"]

ABLATIONS = {
    "features": {
        "full": {},
        "age_only": {"features": {"token": ["age_log", "rel_pos", "pos_log", "is_generated"]}},
        "no_attention": {"features": {"token": [f for f in TOKEN_FEATURES if f not in ATTN]}},
        "no_history": {"features": {"token": [f for f in TOKEN_FEATURES if f not in HIST]}},
        "no_norms": {"features": {"token": [f for f in TOKEN_FEATURES if f not in NORMS]}},
        "attention_only": {"features": {"token": [*ATTN, "age_log"]}},
    },
    "arch": {
        "mlp": {"rl": {"arch": "mlp"}},
        "deepsets": {"rl": {"arch": "deepsets"}},
        "setattn": {"rl": {"arch": "setattn"}},
    },
    "reward": {
        "layer_max_crit": {},  # v1.2 default: layer-max reward + critical-eviction penalty
        "layer_mean": {"sim": {"layer_max_reward": False}},
        "no_crit_penalty": {"sim": {"lambda_crit": 0.0}},
        "no_task_term": {"sim": {"lambda_task": 0.0}},
        "no_privileged_critic": {"rl": {"privileged_critic": False}},
        "shared_advantage": {"rl": {"advantage_mode": "shared"}},
    },
    "sampler": {
        "per_slot": {"rl": {"ratio_mode": "per_slot"}},
        "sequence": {"rl": {"ratio_mode": "sequence"}},
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--only", default=None, help="comma-separated ablation groups")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-warm",
        action="store_true",
        help="train every variant from scratch (fair for feature ablations)",
    )
    args = ap.parse_args()
    groups = args.only.split(",") if args.only else list(ABLATIONS)
    results = []
    for g in groups:
        for name, delta in ABLATIONS[g].items():
            cfg = load_config(args.config)
            cfg["training"]["total_steps"] = args.steps
            cfg["training"]["eval_every_updates"] = 1000  # only the final eval
            cfg["seed"] = args.seed
            cfg["checkpoint_name"] = f"ablate_{g}_{name}"
            if args.no_warm:
                cfg["rl"]["init_from"] = None
            elif "features" in delta:
                # warm-start protocol: train a regressor on the SAME feature subset first
                feats = delta["features"]["token"]
                reg_path = Path(f"checkpoints/ablate_reg_{name}.pt")
                if not reg_path.exists():
                    import subprocess

                    subprocess.run(
                        [
                            sys.executable,
                            "scripts/train_regressor.py",
                            "--out",
                            str(reg_path),
                            "--features",
                            ",".join(feats),
                            "--episodes",
                            "80",
                            "--epochs",
                            "20",
                            "--max-prompt",
                            str(cfg["data"].get("max_prompt") or 100000),
                        ],
                        check=True,
                    )
                cfg["rl"]["init_from"] = str(reg_path)
            # apply delta (shallow per section)
            for sec, vals in delta.items():
                cfg.setdefault(sec, {}).update(vals)
            cfg["ablation"] = {"group": g, "name": name}
            t0 = time.time()
            with start_run(
                "train", cfg, seed=args.seed, device="cpu", tags={"ablation": f"{g}/{name}"}
            ) as run:
                res = train(cfg, run, log=lambda *a: None)
                run.finish(res)
            best = Path(run.dir) / "best_eval.json"
            summ = json.loads(best.read_text()) if best.exists() else []
            rl = {r["budget_frac"]: r["lost_mass_decode"] for r in summ if r["controller"] == "rl"}
            h2o = {
                r["budget_frac"]: r["lost_mass_decode"] for r in summ if r["controller"] == "h2o"
            }
            results.append(
                {
                    "group": g,
                    "name": name,
                    "run_id": run.run_id,
                    "seconds": round(time.time() - t0),
                    "rl_lost_decode": rl,
                    "h2o_lost_decode": h2o,
                    "rl_at_25": rl.get(0.25),
                    "h2o_at_25": h2o.get(0.25),
                }
            )
            print(
                f"[ablate] {g}/{name}: rl@25% {rl.get(0.25)} (h2o {h2o.get(0.25)}) run {run.run_id}",
                flush=True,
            )
    Path("docs").mkdir(exist_ok=True)
    lines = [
        f"# Ablations (sim, {args.steps} decision steps each, seed {args.seed}) — auto-generated\n",
        "| group | variant | rl lost-mass@25% (val, det.) | h2o@25% | run |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['group']} | {r['name']} | {r['rl_at_25']:.4f} | {r['h2o_at_25']:.4f} | `{r['run_id']}` |"
            if r["rl_at_25"] is not None
            else f"| {r['group']} | {r['name']} | – | – | `{r['run_id']}` |"
        )
    Path("docs/ablations.md").write_text("\n".join(lines) + "\n")
    Path("docs/ablations.json").write_text(json.dumps(results, indent=1))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    labels = [f"{r['group']}/{r['name']}" for r in results]
    vals = [r["rl_at_25"] or 0 for r in results]
    ax.barh(labels[::-1], vals[::-1], color="#2ca02c")
    if results and results[0]["h2o_at_25"]:
        ax.axvline(results[0]["h2o_at_25"], color="#1f77b4", ls="--", label="h2o")
        ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel("sim lost attention mass (decode) at 25% budget — lower is better")
    fig.tight_layout()
    Path("docs/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("docs/figures/ablations.png")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
