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
    """Per-slot metadata lives on CPU (controllers run there); only the K/V tensors and the
    stats buffer live on the model device. Norms/cosines for newly appended slots are
    computed lazily, once per decision, to avoid per-token device round-trips."""

    def __init__(self, model: HFCausalLM, n_sink: int = 4):
        self.model = model
        self.info = model.info
        self.device = model.device
        self.cache = model.new_cache()
        self.n_sink = n_sink
        self.positions = torch.zeros(0, dtype=torch.long)
        self.chunk_id = torch.zeros(0, dtype=torch.long)
        self.is_generated = torch.zeros(0, dtype=torch.bool)
        self.attn_cum_mean = torch.zeros(0)
        self.attn_cum_max = torch.zeros(0)
        self.k_norm = torch.zeros(0)
        self.v_norm = torch.zeros(0)
        self.adj_cos = torch.zeros(0)
        self._pending = 0  # appended slots whose norms are not computed yet
        self._last_key: torch.Tensor | None = None  # [L, Hkv, d] last appended key (adj_cos)
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
    def append(self, positions: torch.Tensor, is_generated: bool, step: int) -> None:
        """Register ``q`` new slots after a forward pass appended them to the cache (cheap)."""
        q = int(positions.numel())
        self.positions = torch.cat([self.positions, positions.detach().cpu().long()])
        self.chunk_id = torch.cat([self.chunk_id, torch.full((q,), step, dtype=torch.long)])
        self.is_generated = torch.cat(
            [self.is_generated, torch.full((q,), is_generated, dtype=torch.bool)]
        )
        self._pending += q
        self.total_appended += q

    @torch.no_grad()
    def _materialize_pending(self) -> None:
        """Compute K/V norms + adjacent-key cosine for slots appended since the last call."""
        q = self._pending
        n = self.n
        self.assert_consistent()
        if q == 0:
            return
        s0 = n - q
        keys = torch.stack([layer.keys[0, :, s0:n, :] for layer in self.cache.layers]).float()
        vals = torch.stack([layer.values[0, :, s0:n, :] for layer in self.cache.layers]).float()
        k_norm = keys.norm(dim=-1).mean(dim=(0, 1))  # [q]
        v_norm = vals.norm(dim=-1).mean(dim=(0, 1))
        cos = torch.zeros(q, device=self.device)
        if q > 1:
            cos[1:] = torch.nn.functional.cosine_similarity(
                keys[:, :, 1:, :], keys[:, :, :-1, :], dim=-1
            ).mean(dim=(0, 1))
        if self._last_key is not None:
            cos[0] = torch.nn.functional.cosine_similarity(
                keys[:, :, 0, :], self._last_key, dim=-1
            ).mean()
        self._last_key = keys[:, :, -1, :].clone()
        block = torch.stack([k_norm, v_norm, cos]).cpu()  # one transfer
        self.k_norm = torch.cat([self.k_norm, block[0]])
        self.v_norm = torch.cat([self.v_norm, block[1]])
        self.adj_cos = torch.cat([self.adj_cos, block[2]])
        self.attn_cum_mean = torch.cat([self.attn_cum_mean, torch.zeros(q)])
        self.attn_cum_max = torch.cat([self.attn_cum_max, torch.zeros(q)])
        self._pending = 0

    def chunk_attention(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(layer-mean, layer-max) attention fraction per slot from the stats buffer (CPU)."""
        n = self.n
        frac = self.model.stats.normalized(n)  # [L, n] on device
        both = torch.stack([frac.mean(dim=0), frac.max(dim=0).values]).cpu()
        return both[0], both[1]

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
        """Snapshot for the controller (CPU tensors). If ``accumulate``, fold this chunk's
        attention into the cumulative statistics and reset the stats buffer (once per decision)."""
        self._materialize_pending()
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
        """Physically keep only ``keep_slots`` (any device); returns number evicted."""
        self._materialize_pending()
        n = self.n
        keep = validate_keep(keep_slots.detach().cpu(), n)
        n_evict = n - int(keep.numel())
        if n_evict == 0:
            return 0
        keep_dev = keep.to(self.device)
        compact_cache(self.cache, keep_dev)
        self.model.stats.compact(keep_dev, n)
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
        self.total_evicted += n_evict
        self.assert_consistent()
        return n_evict
