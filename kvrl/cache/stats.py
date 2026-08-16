"""Attention-mass statistics accumulated inside the custom attention function.

``StatsBuffer`` holds, per layer and per *cache slot*, the attention mass received during
the current chunk (summed over heads and over the chunk's queries), plus counters needed to
turn sums into the normalised quantities the controllers/features consume:

    A^l_k(j) = mass[l, j] / (n_heads * n_queries)      (Σ_j A^l_k(j) = 1 per layer)

The buffer is indexed by cache slot (not absolute position) and is compacted alongside the
KV tensors by :func:`kvrl.cache.compact.compact_cache`.
"""

from __future__ import annotations

import torch


class StatsBuffer:
    def __init__(self, n_layers: int, n_heads: int, max_slots: int, device: torch.device | str):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_slots = max_slots
        self.device = torch.device(device)
        self.mass = torch.zeros(n_layers, max_slots, dtype=torch.float32, device=self.device)
        self.n_queries = 0  # queries accumulated since last reset (same for every layer)
        self.enabled = False

    # -- accumulation (called from the attention function) --------------------------------
    def add(self, layer_idx: int, mass_per_slot: torch.Tensor) -> None:
        kv = mass_per_slot.shape[-1]
        if kv > self.max_slots:
            self.grow(kv)
        self.mass[layer_idx, :kv] += mass_per_slot.to(self.mass.dtype)

    def note_queries(self, n: int) -> None:
        self.n_queries += n

    def grow(self, min_slots: int) -> None:
        new_max = max(min_slots, self.max_slots * 2)
        new = torch.zeros(self.n_layers, new_max, dtype=torch.float32, device=self.device)
        new[:, : self.max_slots] = self.mass
        self.mass = new
        self.max_slots = new_max

    # -- consumption ---------------------------------------------------------------------
    def normalized(self, kv_len: int) -> torch.Tensor:
        """[n_layers, kv_len] attention fraction per slot for the accumulated queries."""
        if kv_len > self.max_slots:  # stats disabled -> buffer never grew via add() (BUG-002)
            self.grow(kv_len)
        denom = max(1, self.n_heads * self.n_queries)
        return self.mass[:, :kv_len] / denom

    def reset(self) -> None:
        self.mass.zero_()
        self.n_queries = 0

    def compact(self, keep_slots: torch.Tensor, kv_len: int) -> None:
        """Keep only ``keep_slots`` (sorted LongTensor over the first ``kv_len`` slots)."""
        if kv_len > self.max_slots:
            self.grow(kv_len)
        k = keep_slots.to(self.device)
        n = k.numel()
        kept = self.mass[:, :kv_len].index_select(1, k)
        self.mass[:, :n] = kept
        self.mass[:, n:kv_len] = 0.0

    def to(self, device: torch.device | str) -> StatsBuffer:
        self.device = torch.device(device)
        self.mass = self.mass.to(self.device)
        return self
