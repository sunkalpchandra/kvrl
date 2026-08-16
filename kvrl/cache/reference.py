"""Masked-reference oracle: express eviction as a mask over the FULL cache.

Used only in tests / validation to prove that physical compaction is numerically equivalent
to hiding the same tokens with an attention mask (identical up to floating point).
"""

from __future__ import annotations

import torch

from kvrl.models.hf_model import HFCausalLM


class MaskedReference:
    """Full-cache run where 'evicted' tokens are masked out instead of removed."""

    def __init__(self, model: HFCausalLM):
        self.model = model
        self.cache = model.new_cache()
        self.alive = torch.zeros(0, dtype=torch.bool, device=model.device)

    @property
    def n_full(self) -> int:
        return int(self.alive.numel())

    def evict_positions(self, dead_positions: torch.Tensor) -> None:
        """Mark absolute positions as evicted (never physically removed)."""
        self.alive[dead_positions.to(self.alive.device)] = False

    @torch.no_grad()
    def forward_chunk(
        self, input_ids: torch.Tensor, positions: torch.Tensor, logits_to_keep: int = 1
    ) -> torch.Tensor:
        q = int(input_ids.numel())
        kv = self.n_full + q
        alive = torch.cat([self.alive, torch.ones(q, dtype=torch.bool, device=self.alive.device)])
        qi = torch.arange(q, device=alive.device)[:, None]
        kj = torch.arange(kv, device=alive.device)[None, :]
        causal = kj <= (kv - q) + qi
        mask = (causal & alive[None, :])[None, None]  # [1,1,q,kv]
        out = self.model.model(
            input_ids=input_ids.reshape(1, -1).to(self.model.device),
            position_ids=positions.reshape(1, -1).to(self.model.device),
            past_key_values=self.cache,
            attention_mask=mask,
            use_cache=True,
            logits_to_keep=logits_to_keep,
        )
        self.alive = alive
        return out.logits
