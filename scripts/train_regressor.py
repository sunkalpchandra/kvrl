#!/usr/bin/env python
"""Supervised baseline: predict discounted future attention mass from the policy's own features.

Rollouts under mixed heuristics (h2o, random, window) on the training traces produce
(obs, F^γ) pairs for candidate tokens; an MLP with the same architecture as the RL policy
is fit with MSE on log1p(F^γ·n). Its output (negated) is used as an eviction score by
`RegressorController` — the "learn a good score, then top-k" baseline RL must beat.

    python scripts/train_regressor.py --train data/raw/train --out checkpoints/regressor_v1.pt
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvrl.controllers import make_controller
from kvrl.controllers.learned import save_policy_checkpoint
from kvrl.features import GLOBAL_FEATURES, FeatureConfig
from kvrl.rl.policy import ScorePolicy
from kvrl.rl.train import feature_norm_constants, load_traces
from kvrl.sim.env import CacheSimEnv
from kvrl.utils.seed import seed_everything


def collect_pairs(traces, fcfg, budget_fracs, n_episodes, seed=0, max_rows=400_000):
    rng = random.Random(seed)
    env = CacheSimEnv(gamma=0.99, feature_cfg=fcfg)
    X, G, Y = [], [], []
    n_rows = 0
    for ep in range(n_episodes):
        tr = rng.choice(traces)
        bf = rng.choice(budget_fracs)
        name = rng.choice(["h2o", "random", "window", "snapkv"])
        ctrl = make_controller(name, seed=ep) if name == "random" else make_controller(name)
        ctrl.reset(episode=ep)
        res = env.reset(tr, budget_frac=bf)
        while not res.done:
            st = env.state
            fut = env.future_mass()  # [n] privileged target
            cand = res.cand_mask
            n = st.n
            # subsample candidates to keep the dataset balanced across steps
            idx = torch.nonzero(cand).flatten()
            if idx.numel() > 256:
                idx = idx[torch.randperm(idx.numel())[:256]]
            X.append(res.obs_tok[idx].numpy().astype(np.float32))
            G.append(np.repeat(res.obs_glob.numpy()[None], idx.numel(), axis=0).astype(np.float32))
            Y.append(np.log1p(fut[idx].numpy() * n).astype(np.float32))
            n_rows += idx.numel()
            keep = ctrl.decide(st, env.budget)
            ctrl.on_compact(keep, n)
            res = env.step(env.keep_to_evict(keep))
        if n_rows >= max_rows:
            break
    return np.concatenate(X), np.concatenate(G), np.concatenate(Y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/raw/train")
    ap.add_argument("--val", default="data/raw/val")
    ap.add_argument("--out", default="checkpoints/regressor_v1.pt")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", default=None, help="comma-separated token-feature subset")
    ap.add_argument("--max-prompt", type=int, default=None)
    args = ap.parse_args()
    seed_everything(args.seed)
    traces = load_traces(args.train, max_prompt=args.max_prompt)
    val_traces = load_traces(args.val, max_prompt=args.max_prompt)
    feats = args.features.split(",") if args.features else None
    fcfg = FeatureConfig(
        **feature_norm_constants(traces), **({"token_features": feats} if feats else {})
    )
    t0 = time.time()
    X, G, Y = collect_pairs(traces, fcfg, [0.125, 0.25, 0.5], args.episodes, seed=args.seed)
    print(
        f"[regressor] {X.shape[0]} rows from {args.episodes} episodes in {time.time() - t0:.0f}s; "
        f"target mean {Y.mean():.3f} std {Y.std():.3f}",
        flush=True,
    )
    Xv = Gv = Yv = None
    if val_traces:
        Xv, Gv, Yv = collect_pairs(val_traces, fcfg, [0.25], 20, seed=args.seed + 1)
    net = ScorePolicy(len(fcfg.token_features), len(GLOBAL_FEATURES), arch="mlp")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt, Gt, Yt = torch.from_numpy(X), torch.from_numpy(G), torch.from_numpy(Y)
    n = Xt.shape[0]
    bs = 4096
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, bs):
            i = perm[s : s + bs]
            tok, glob = Xt[i][:, None, :], Gt[i]
            pred = -net(tok, glob, torch.ones(i.numel(), 1, dtype=torch.bool))[:, 0]
            loss = torch.nn.functional.mse_loss(pred, Yt[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * i.numel()
        msg = f"[regressor] epoch {ep + 1} train mse {tot / n:.4f}"
        if Xv is not None:
            with torch.no_grad():
                pv = -net(
                    torch.from_numpy(Xv)[:, None, :],
                    torch.from_numpy(Gv),
                    torch.ones(Xv.shape[0], 1, dtype=torch.bool),
                )[:, 0]
                yv = torch.from_numpy(Yv)
                mse = torch.nn.functional.mse_loss(pv, yv).item()
                corr = torch.corrcoef(torch.stack([pv, yv]))[0, 1].item()
            msg += f" | val mse {mse:.4f} corr {corr:.3f}"
        print(msg, flush=True)
    save_policy_checkpoint(
        args.out,
        net,
        fcfg,
        kind="regressor",
        meta={"rows": int(n), "episodes": args.episodes, "epochs": args.epochs, "seed": args.seed},
    )
    print(f"[regressor] saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
