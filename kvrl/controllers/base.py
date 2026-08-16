"""Controller contract shared by heuristics, oracles and the RL policy.

``decide(state, budget)`` returns the sorted keep-set (cache slots) of size ≤ budget.
Most controllers only need to produce a per-slot *keep score*; :func:`select_keep` turns
scores into a keep-set that always contains the protected slots (sinks + current chunk).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from kvrl.cache.view import CacheState


def select_keep(scores: torch.Tensor, state: CacheState, budget: int) -> torch.Tensor:
    """Keep protected slots plus the highest-scoring candidates up to ``budget`` slots."""
    n = state.n
    if budget >= n:
        return torch.arange(n, device=scores.device)
    protected = state.protected_mask().to(scores.device)
    n_prot = int(protected.sum())
    room = budget - n_prot
    keep = protected.clone()
    if room > 0:
        cand_scores = scores.masked_fill(protected, float("-inf"))
        top = torch.topk(cand_scores, k=room).indices
        keep[top] = True
    # if protected alone exceed budget, we still keep them (reported as over-budget by engine)
    return torch.nonzero(keep, as_tuple=False).flatten().sort().values


class KVCacheController(ABC):
    name: str = "base"
    #: whether the controller needs attention statistics (drives the stats path on/off)
    needs_attention: bool = False

    def reset(self, **episode_info) -> None:  # noqa: B027 - optional hook
        """Called at the start of every prompt/episode."""

    def observe(self, state: CacheState, budget: int) -> None:  # noqa: B027
        """Called on EVERY decision step (also when nothing must be evicted), before decide."""

    @abstractmethod
    def decide(self, state: CacheState, budget: int) -> torch.Tensor: ...

    def on_compact(self, keep_slots: torch.Tensor, n_before: int) -> None:  # noqa: B027
        """Called after the engine applied a keep-set (for controllers with per-slot state)."""

    def describe(self) -> dict:
        return {"name": self.name}


class ScoreController(KVCacheController):
    """Base for controllers expressed as a per-slot keep score (higher = keep)."""

    def scores(self, state: CacheState) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def decide(self, state: CacheState, budget: int) -> torch.Tensor:
        if budget >= state.n:
            return torch.arange(state.n, device=state.positions.device)
        s = self.scores(state)
        return select_keep(s.float(), state, budget)

    #: optional per-slot importance for visualisation (defaults to the score)
    def importance(self, state: CacheState) -> torch.Tensor:
        return self.scores(state).float()
