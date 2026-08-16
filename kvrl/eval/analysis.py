"""Failure analysis in the simulator (cheap, exact bookkeeping).

For a set of traces × budgets × controllers we record, per episode:
- critical-token retention when the question chunk is processed and during decode
- age of tokens at eviction (evict-age histogram: "window in disguise" / "inverse window")
- retention profile by relative position (which parts of the context survive)
- lost attention mass on decode steps
and, for a learned controller, permutation importance of each observation feature
(shuffle one feature across tokens at every decision → Δ lost mass).
"""

from __future__ import annotations

import numpy as np
import torch

from kvrl.controllers.learned import RLController
from kvrl.features import TOKEN_FEATURES
from kvrl.sim.env import CacheSimEnv
from kvrl.traces.storage import Trace


def analyse_episode(
    env: CacheSimEnv, trace: Trace, controller, budget_frac: float, n_bins: int = 20
) -> dict:
    res = env.reset(trace, budget_frac=budget_frac)
    controller.reset(
        episode=0, n_prompt=trace.n_prompt, budget=env.budget, max_new_tokens=env.max_new_tokens
    )
    evict_ages: list[int] = []
    crit = trace.critical_mask
    crit_at_question = None
    while not res.done:
        st = env.state
        controller.observe(st, env.budget)
        keep = controller.decide(st, env.budget)
        ev = env.keep_to_evict(keep)
        ages = (st.step - st.chunk_id[ev]).tolist()
        evict_ages.extend(ages)
        controller.on_compact(keep, st.n)
        # last prefill decision = question chunk processed
        if st.phase == 0 and crit.any():
            alive_after = env.alive[keep.numpy()]
            crit_at_question = float(crit[alive_after].sum() / crit.sum())
        res = env.step(ev)
    alive = np.zeros(trace.T, dtype=bool)
    alive[env.alive] = True
    # retention profile by relative position over the prompt
    pos = np.arange(trace.n_prompt)
    bins = np.minimum((pos / max(1, trace.n_prompt) * n_bins).astype(int), n_bins - 1)
    profile = np.array(
        [
            alive[: trace.n_prompt][bins == b].mean() if (bins == b).any() else np.nan
            for b in range(n_bins)
        ]
    )
    m = res.info
    return {
        "trace_id": trace.trace_id,
        "task": trace.meta.get("task"),
        "budget_frac": budget_frac,
        "controller": controller.name,
        "lost_mass_decode": m["lost_mass_decode"],
        "lost_mass_mean": m["lost_mass_mean"],
        "crit_retained_decode": m["crit_retained"],
        "crit_retained_at_question": crit_at_question,
        "n_critical": int(crit.sum()),
        "n_evictions": m["n_evictions"],
        "evict_age_mean": float(np.mean(evict_ages)) if evict_ages else float("nan"),
        "evict_age_hist": np.histogram(evict_ages, bins=[0, 1, 2, 4, 8, 16, 32, 64, 128, 1024])[
            0
        ].tolist(),
        "retention_profile": profile.tolist(),
    }


class _PermutedRL(RLController):
    """RL controller with one observation feature shuffled across tokens at every decision."""

    def __init__(self, base: RLController, feature_idx: int, seed: int = 0):
        super().__init__(base.policy, base.feature_cfg, deterministic=True)
        self.feature_idx = feature_idx
        self.gen = torch.Generator().manual_seed(seed)
        self.name = f"rl_perm_{TOKEN_FEATURES[feature_idx]}"

    def observe(self, state, budget):
        super().observe(state, budget)
        tok, glob = self._obs
        perm = torch.randperm(tok.shape[0], generator=self.gen)
        tok = tok.clone()
        tok[:, self.feature_idx] = tok[perm, self.feature_idx]
        self._obs = (tok, glob)


def permutation_importance(
    env: CacheSimEnv,
    traces: list[Trace],
    rl: RLController,
    budget_frac: float,
    features: list[str] | None = None,
) -> list[dict]:
    """Δ lost-mass (decode) when each feature is shuffled; larger = more relied upon."""
    feats = features or list(rl.feature_cfg.token_features)
    base = [
        analyse_episode(env, t, RLController(rl.policy, rl.feature_cfg), budget_frac)[
            "lost_mass_decode"
        ]
        for t in traces
    ]
    out = []
    for f in feats:
        idx = rl.feature_cfg.token_features.index(f)
        vals = [
            analyse_episode(env, t, _PermutedRL(rl, idx), budget_frac)["lost_mass_decode"]
            for t in traces
        ]
        out.append(
            {
                "feature": f,
                "lost_mass_base": float(np.mean(base)),
                "lost_mass_permuted": float(np.mean(vals)),
                "delta": float(np.mean(vals) - np.mean(base)),
            }
        )
    return sorted(out, key=lambda r: -r["delta"])
