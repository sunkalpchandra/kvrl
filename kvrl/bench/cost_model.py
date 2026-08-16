"""Hardware cost model fitted from measurements (never assumed).

Fits decode ms/token as an affine function of KV cache length from the benchmark's
decode-vs-cache-length curve (``runs/<bench>/results.json``), so controllers/budgets can be
compared on *estimated decode cost* at any cache size, and reports the fit quality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DecodeCostModel:
    ms_per_token_base: float  # intercept: cost at cache length 0
    ms_per_token_per_1k: float  # slope per 1024 cached tokens
    r2: float
    n_points: int
    device: str = ""

    def decode_ms(self, cache_len: float) -> float:
        return self.ms_per_token_base + self.ms_per_token_per_1k * cache_len / 1024.0

    def as_dict(self) -> dict:
        return {
            "ms_per_token_base": self.ms_per_token_base,
            "ms_per_token_per_1k": self.ms_per_token_per_1k,
            "r2": self.r2,
            "n_points": self.n_points,
            "device": self.device,
        }


def fit_decode_cost(curve: list[dict], device: str = "") -> DecodeCostModel:
    """curve rows: {cache_len, decode_ms_per_tok_median}."""
    x = np.array([r["cache_len"] for r in curve], dtype=float) / 1024.0
    y = np.array([r["decode_ms_per_tok_median"] for r in curve], dtype=float)
    if len(x) < 2:
        raise ValueError("need at least two curve points")
    A = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-9
    return DecodeCostModel(float(coef[0]), float(coef[1]), 1 - ss_res / ss_tot, len(x), device)
