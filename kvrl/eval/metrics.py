"""Quality metrics and statistics for controller comparisons.

- accuracy on retrieval-style tasks (see :func:`kvrl.eval.tasks.is_correct`)
- output fidelity between a budgeted generation and the full-cache generation
  (token-level agreement prefix + ROUGE-L), because the goal is *preserving the model's
  behaviour*, not matching a human reference
- NLL delta of a fixed continuation (language-modelling degradation)
- bootstrap confidence intervals and paired differences per prompt
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def lcs_length(a: Sequence, b: Sequence) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: Sequence, ref: Sequence, beta: float = 1.2) -> float:
    """ROUGE-L F-score over token sequences (0..1)."""
    l = lcs_length(pred, ref)
    if l == 0:
        return 0.0
    p, r = l / len(pred), l / len(ref)
    return (1 + beta**2) * p * r / (r + beta**2 * p)


def prefix_agreement(pred: Sequence, ref: Sequence) -> float:
    """Fraction of ``ref`` reproduced exactly as a prefix (greedy decoding divergence point)."""
    if not ref:
        return 1.0
    n = 0
    for x, y in zip(pred, ref):
        if x != y:
            break
        n += 1
    return n / len(ref)


def token_agreement(pred: Sequence, ref: Sequence) -> float:
    """Position-wise agreement over the reference length."""
    if not ref:
        return 1.0
    return sum(1 for x, y in zip(pred, ref) if x == y) / len(ref)


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap CI of the mean."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (math.nan, math.nan, math.nan)
    if v.size == 1:
        return (float(v[0]), float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return (
        float(v.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def paired_difference(
    a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 0
) -> dict:
    """Paired comparison a - b over the same prompts: mean diff, CI, win rate."""
    a_, b_ = np.asarray(a, float), np.asarray(b, float)
    if a_.shape != b_.shape:
        raise ValueError("paired arrays must have the same shape")
    d = a_ - b_
    mean, lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    return {
        "mean_diff": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "win_rate": float(np.mean(d > 0)) if d.size else math.nan,
        "tie_rate": float(np.mean(d == 0)) if d.size else math.nan,
        "n": int(d.size),
        "significant": bool(d.size and (lo > 0 or hi < 0)),
    }


def summarize(values: Sequence[float]) -> dict:
    v = np.asarray(values, float)
    if v.size == 0:
        return {"n": 0}
    mean, lo, hi = bootstrap_ci(v)
    return {
        "n": int(v.size),
        "mean": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "median": float(np.median(v)),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
    }
