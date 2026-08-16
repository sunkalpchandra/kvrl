"""KVCacheView: a DynamicCache plus the per-slot bookkeeping controllers need.

Slots are positions inside the (compacted) cache tensors; ``positions`` maps each slot to
its absolute token position. Every array here is compacted with the same keep-set as the
K/V tensors, so slot ``i`` always refers to the same token everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from kvrl.models.hf_model import HFCausalLM

from .compact import cache_lengths, compact_cache, validate_keep


@dataclass
class CacheState:
    """Everything a controller may look at (identical contract in the simulator)."""

    positions: torch.Tensor  # long [n]  absolute positions
    chunk_id: torch.Tensor  # long [n]  decision step at which the token entered
    is_generated: torch.Tensor  # bool [n]
    attn_last_mean: torch.Tensor  # float [n]  A_k(j): last-chunk attention fraction, layer-mean
    attn_last_max: torch.Tensor  # float [n]  layer-max of the head-mean fraction
    attn_cum_mean: torch.Tensor  # float [n]  Σ over past chunks of attn_last_mean (H2O score)
    attn_cum_max: torch.Tensor  # float [n]  Σ over past chunks of attn_last_max
    k_norm: torch.Tensor  # float [n]
    v_norm: torch.Tensor  # float [n]
    adj_cos: torch.Tensor  # float [n]  cosine(key_j, key_{j-1}) at insertion (layer/head-mean)
    n_new: int  # tokens appended since the previous decision (protected window)
    step: int  # decision index k
    ctx_len: int  # true number of tokens processed so far (absolute position of next token)
    phase: int  # 0 = prefill, 1 = decode
    n_prompt: int
    n_generated: int
    max_new_tokens: int
    n_sink: int = 4
    extras: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.positions.numel())

    def protected_mask(self) -> torch.Tensor:
        """Sinks (first ``n_sink`` absolute positions) + the current chunk are never evicted."""
        m = self.positions < self.n_sink
        if self.n_new > 0:
            m[-self.n_new :] = True
        return m

    def to(self, device) -> CacheState:
        kw = {}
        for k, v in self.__dict__.items():
            kw[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        return CacheState(**kw)


class KVCacheView:
    def __init__(self, model: HFCausalLM, n_sink: int = 4):
        self.model = model
        self.info = model.info
        self.device = model.device
        self.cache = model.new_cache()
        self.n_sink = n_sink
        z = torch.zeros(0, device=self.device)
        self.positions = torch.zeros(0, dtype=torch.long, device=self.device)
        self.chunk_id = torch.zeros(0, dtype=torch.long, device=self.device)
        self.is_generated = torch.zeros(0, dtype=torch.bool, device=self.device)
        self.attn_cum_mean = z.clone()
        self.attn_cum_max = z.clone()
        self.k_norm = z.clone()
        self.v_norm = z.clone()
        self.adj_cos = z.clone()
        self._last_key: torch.Tensor | None = None  # [L, Hkv, d] last appended key (for adj_cos)
        self.total_evicted = 0
        self.total_appended = 0

    # ------------------------------------------------------------------ properties
    @property
    def n(self) -> int:
        return int(self.positions.numel())

    def assert_consistent(self) -> None:
        lens = cache_lengths(self.cache)
        assert all(l == self.n for l in lens), f"cache lengths {lens} != view {self.n}"

    def kv_bytes(self) -> int:
        return self.info.kv_bytes(self.n)

    # ------------------------------------------------------------------ append / stats
    @torch.no_grad()
    def append(self, positions: torch.Tensor, is_generated: bool, step: int) -> None:
        """Register ``q`` new slots after a forward pass appended them to the cache."""
        q = int(positions.numel())
        n_before = self.n
        self.assert_consistent_after_append(n_before + q)
        keys = torch.stack(
            [layer.keys[0, :, n_before : n_before + q, :] for layer in self.cache.layers]
        )
        vals = torch.stack(
            [layer.values[0, :, n_before : n_before + q, :] for layer in self.cache.layers]
        )
        # [L, Hkv, q, d] -> norms averaged over layers and kv heads
        k_norm = keys.float().norm(dim=-1).mean(dim=(0, 1))
        v_norm = vals.float().norm(dim=-1).mean(dim=(0, 1))
        # adjacent-key cosine within the new block (+ against the last previously appended key)
        kf = keys.float()
        prev = self._last_key
        cos = torch.zeros(q, device=self.device)
        if q > 1:
            a, b = kf[:, :, 1:, :], kf[:, :, :-1, :]
            cos[1:] = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean(dim=(0, 1))
        if prev is not None:
            cos[0] = torch.nn.functional.cosine_similarity(kf[:, :, 0, :], prev, dim=-1).mean()
        self._last_key = kf[:, :, -1, :].clone()
        dev = self.device
        self.positions = torch.cat([self.positions, positions.to(dev)])
        self.chunk_id = torch.cat(
            [self.chunk_id, torch.full((q,), step, dtype=torch.long, device=dev)]
        )
        self.is_generated = torch.cat(
            [self.is_generated, torch.full((q,), is_generated, dtype=torch.bool, device=dev)]
        )
        self.attn_cum_mean = torch.cat([self.attn_cum_mean, torch.zeros(q, device=dev)])
        self.attn_cum_max = torch.cat([self.attn_cum_max, torch.zeros(q, device=dev)])
        self.k_norm = torch.cat([self.k_norm, k_norm])
        self.v_norm = torch.cat([self.v_norm, v_norm])
        self.adj_cos = torch.cat([self.adj_cos, cos])
        self.total_appended += q

    def assert_consistent_after_append(self, expected: int) -> None:
        lens = cache_lengths(self.cache)
        assert all(l == expected for l in lens), f"cache lengths {lens} != expected {expected}"

    def chunk_attention(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(layer-mean, layer-max) attention fraction per slot from the stats buffer."""
        n = self.n
        frac = self.model.stats.normalized(n)  # [L, n]
        return frac.mean(dim=0), frac.max(dim=0).values

    def state(
        self,
        *,
        n_new: int,
        step: int,
        ctx_len: int,
        phase: int,
        n_prompt: int,
        n_generated: int,
        max_new_tokens: int,
        accumulate: bool = True,
    ) -> CacheState:
        """Snapshot for the controller. If ``accumulate``, fold this chunk's attention into
        the cumulative statistics and reset the stats buffer (call once per decision)."""
        mean, lmax = self.chunk_attention()
        if accumulate:
            self.attn_cum_mean = self.attn_cum_mean + mean
            self.attn_cum_max = self.attn_cum_max + lmax
            self.model.stats.reset()
        return CacheState(
            positions=self.positions,
            chunk_id=self.chunk_id,
            is_generated=self.is_generated,
            attn_last_mean=mean,
            attn_last_max=lmax,
            attn_cum_mean=self.attn_cum_mean.clone(),
            attn_cum_max=self.attn_cum_max.clone(),
            k_norm=self.k_norm,
            v_norm=self.v_norm,
            adj_cos=self.adj_cos,
            n_new=n_new,
            step=step,
            ctx_len=ctx_len,
            phase=phase,
            n_prompt=n_prompt,
            n_generated=n_generated,
            max_new_tokens=max_new_tokens,
            n_sink=self.n_sink,
        )

    # ------------------------------------------------------------------ eviction
    @torch.no_grad()
    def compact(self, keep_slots: torch.Tensor) -> int:
        """Physically keep only ``keep_slots``; returns number evicted."""
        n = self.n
        keep = validate_keep(keep_slots, n).to(self.device)
        n_evict = n - int(keep.numel())
        if n_evict == 0:
            return 0
        compact_cache(self.cache, keep)
        for name in (
            "positions",
            "chunk_id",
            "is_generated",
            "attn_cum_mean",
            "attn_cum_max",
            "k_norm",
            "v_norm",
            "adj_cos",
        ):
            setattr(self, name, getattr(self, name).index_select(0, keep))
        self.model.stats.compact(keep, n)
        self.total_evicted += n_evict
        self.assert_consistent()
        return n_evict
