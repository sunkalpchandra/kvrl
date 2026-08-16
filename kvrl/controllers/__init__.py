"""KV cache controllers: one interface for heuristics, oracles and the RL policy."""

from .base import KVCacheController, ScoreController, select_keep
from .heuristics import HEURISTICS

__all__ = ["HEURISTICS", "KVCacheController", "ScoreController", "select_keep"]


def make_controller(name: str, **kw) -> KVCacheController:
    """Factory: heuristics by name; ``rl`` is resolved lazily to avoid importing torch RL code."""
    if name in HEURISTICS:
        return HEURISTICS[name](**kw)
    if name in ("rl", "regressor"):
        from .learned import make_learned_controller

        return make_learned_controller(name, **kw)
    raise KeyError(
        f"unknown controller {name!r}; known: {sorted(HEURISTICS)} + ['rl', 'regressor']"
    )
