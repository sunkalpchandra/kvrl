"""Non-learned baselines. All share the protected-slot rule (sinks + current chunk)."""

from __future__ import annotations

import torch

from kvrl.cache.view import CacheState

from .base import KVCacheController, ScoreController


class FullCacheController(KVCacheController):
    """Keep everything (ignores the budget). The reference for quality."""

    name = "full"

    def decide(self, state: CacheState, budget: int) -> torch.Tensor:
        return torch.arange(state.n, device=state.positions.device)


class SlidingWindowController(ScoreController):
    """StreamingLLM-style: sinks + the most recent tokens (score = position)."""

    name = "window"

    def scores(self, state: CacheState) -> torch.Tensor:
        return state.positions.float()


class RandomController(ScoreController):
    """Random eviction among candidates (seeded)."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.gen = torch.Generator().manual_seed(seed)

    def reset(self, **episode_info) -> None:
        self.gen = torch.Generator().manual_seed(self.seed + int(episode_info.get("episode", 0)))

    def scores(self, state: CacheState) -> torch.Tensor:
        return torch.rand(state.n, generator=self.gen).to(state.positions.device)


class LastChunkAttentionController(ScoreController):
    """SnapKV-style: keep the tokens most attended by the most recent chunk of queries."""

    name = "snapkv"
    needs_attention = True

    def scores(self, state: CacheState) -> torch.Tensor:
        return state.attn_last_mean


class HeavyHitterController(ScoreController):
    """H2O-style: cumulative attention (heavy hitters) + a recent window.

    ``recent_frac`` of the budget is reserved for the most recent tokens; the rest goes to
    the tokens with the highest cumulative attention received.
    """

    name = "h2o"
    needs_attention = True

    def __init__(self, recent_frac: float = 0.5, use_layer_max: bool = False):
        self.recent_frac = recent_frac
        self.use_layer_max = use_layer_max
        self._budget = None

    def decide(self, state: CacheState, budget: int) -> torch.Tensor:
        self._budget = budget
        return super().decide(state, budget)

    def scores(self, state: CacheState) -> torch.Tensor:
        cum = state.attn_cum_max if self.use_layer_max else state.attn_cum_mean
        n = state.n
        budget = self._budget if self._budget is not None else n
        n_recent = round(self.recent_frac * budget)
        s = cum.clone().float()
        if n_recent > 0:
            # recent tokens get +inf so they are kept regardless of attention
            order = torch.argsort(state.positions, descending=True)
            s[order[:n_recent]] = float("inf")
        return s


class TOVAController(ScoreController):
    """TOVA-style: keep tokens most attended by the *last* query only (approximated here by
    the last chunk's layer-max mass, the closest we track cheaply)."""

    name = "tova"
    needs_attention = True

    def scores(self, state: CacheState) -> torch.Tensor:
        return state.attn_last_max


class KeyNormController(ScoreController):
    """Evict tokens with the largest key norm (Devoto et al. 2024: low ‖k‖ ↔ high attention)."""

    name = "keynorm"

    def scores(self, state: CacheState) -> torch.Tensor:
        return -state.k_norm


class HybridController(ScoreController):
    """Recency + attention: score = α·norm(cumulative attention) + (1-α)·recency."""

    name = "hybrid"
    needs_attention = True

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def scores(self, state: CacheState) -> torch.Tensor:
        cum = state.attn_cum_mean
        a = cum / (cum.max() + 1e-9)
        pos = state.positions.float()
        r = (pos - pos.min()) / (pos.max() - pos.min() + 1e-9)
        return self.alpha * a + (1 - self.alpha) * r


HEURISTICS: dict[str, type[KVCacheController]] = {
    "full": FullCacheController,
    "window": SlidingWindowController,
    "random": RandomController,
    "snapkv": LastChunkAttentionController,
    "h2o": HeavyHitterController,
    "tova": TOVAController,
    "keynorm": KeyNormController,
    "hybrid": HybridController,
}
