"""FeatureState — incremental per-slot features for the cache controller.

Both environments feed the same :class:`kvrl.cache.view.CacheState` snapshots (real
inference builds them from the live stats buffer; the simulator builds them from recorded
traces) and get back the observation the policy sees:

    tok  : float32 [n, 18]   per cache slot
    glob : float32 [8]       per decision

Attention quantities go through the length-invariant transform
``phi(a; n) = log1p(a * n) / log1p(N_MAX)`` (a·n is the ratio to uniform attention over the
n visible keys, so "uniform" ≈ 0.07 at any context length).

The state must be compacted with the same keep-set as the cache (:meth:`compact`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from kvrl.cache.view import CacheState

N_MAX = 32768.0
_LOG_NMAX = math.log1p(N_MAX)
_LOG_512 = math.log1p(512.0)

TOKEN_FEATURES = [
    "age_log",
    "rel_pos",
    "pos_log",
    "is_generated",
    "attn_last",
    "attn_ema_fast",
    "attn_ema_slow",
    "attn_mean",
    "attn_max",
    "attn_lastmax_layer",
    "attn_disp",
    "hit_rate",
    "since_hit",
    "chunk_share",
    "key_norm",
    "value_norm",
    "adj_key_cos",
    "nbr_evicted",
]
GLOBAL_FEATURES = [
    "budget_frac",
    "occupancy",
    "evict_frac",
    "ctx_log",
    "step_frac",
    "phase",
    "remaining_gen",
    "chunk_entropy",
]


def phi(a: torch.Tensor, n: int | torch.Tensor) -> torch.Tensor:
    return torch.log1p(a.clamp_min(0) * n) / _LOG_NMAX


@dataclass
class FeatureConfig:
    ema_fast: float = 0.5
    ema_slow: float = 0.9
    hit_top_frac: float = 0.05
    k_norm_mean: float = 0.0
    k_norm_std: float = 1.0
    v_norm_mean: float = 0.0
    v_norm_std: float = 1.0
    #: which token features to expose (ablations); default all
    token_features: list[str] = field(default_factory=lambda: list(TOKEN_FEATURES))

    def token_index(self) -> torch.Tensor:
        return torch.tensor([TOKEN_FEATURES.index(f) for f in self.token_features])


class FeatureState:
    def __init__(self, cfg: FeatureConfig | None = None, device: torch.device | str = "cpu"):
        self.cfg = cfg or FeatureConfig()
        self.device = torch.device(device)
        self._sel = self.cfg.token_index().to(self.device)
        self.reset()

    # ---------------------------------------------------------------- lifecycle
    def reset(self) -> None:
        d = self.device
        z = torch.zeros(0, device=d)
        self.n = 0
        self.ema_fast = z.clone()
        self.ema_slow = z.clone()
        self.sum_phi = z.clone()
        self.sum_a = z.clone()
        self.max_phi = z.clone()
        self.hits = z.clone()
        self.last_hit = torch.zeros(0, dtype=torch.long, device=d)  # step of last hit (-1 none)
        self.age_steps = z.clone()  # number of chunk-updates the token has seen
        self.chunk_mass = {}  # chunk_id -> cumulative attention mass (layer-mean) of that chunk
        self.chunk_size0 = {}  # chunk_id -> original size
        self.chunk_alive = {}  # chunk_id -> currently retained count
        self.last_tok: torch.Tensor | None = None
        self.last_glob: torch.Tensor | None = None

    def _grow(self, q: int, step: int) -> None:
        d = self.device
        self.ema_fast = torch.cat([self.ema_fast, torch.zeros(q, device=d)])
        self.ema_slow = torch.cat([self.ema_slow, torch.zeros(q, device=d)])
        self.sum_phi = torch.cat([self.sum_phi, torch.zeros(q, device=d)])
        self.sum_a = torch.cat([self.sum_a, torch.zeros(q, device=d)])
        self.max_phi = torch.cat([self.max_phi, torch.zeros(q, device=d)])
        self.hits = torch.cat([self.hits, torch.zeros(q, device=d)])
        self.last_hit = torch.cat([self.last_hit, torch.full((q,), -1, dtype=torch.long, device=d)])
        self.age_steps = torch.cat([self.age_steps, torch.zeros(q, device=d)])
        self.n += q

    def compact(
        self, keep_slots: torch.Tensor, chunk_ids_before: torch.Tensor | None = None
    ) -> None:
        keep = keep_slots.to(self.device)
        if chunk_ids_before is not None and keep.numel() < self.n:
            evicted = torch.ones(self.n, dtype=torch.bool, device=self.device)
            evicted[keep] = False
            for cid in chunk_ids_before[evicted].tolist():
                self.chunk_alive[cid] = self.chunk_alive.get(cid, 0) - 1
        for name in (
            "ema_fast",
            "ema_slow",
            "sum_phi",
            "sum_a",
            "max_phi",
            "hits",
            "last_hit",
            "age_steps",
        ):
            setattr(self, name, getattr(self, name).index_select(0, keep))
        self.n = int(keep.numel())

    # ---------------------------------------------------------------- update
    def update(
        self, st: CacheState, budget: int, n_decisions_total: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold the chunk statistics in ``st`` into the running state; return (tok, glob)."""
        dev = self.device
        n = st.n
        if n > self.n:
            self._grow(n - self.n, st.step)
        assert n == self.n, f"feature state has {self.n} slots, cache has {n}"
        a = st.attn_last_mean.to(dev).float()
        amax = st.attn_last_max.to(dev).float()
        p = phi(a, n)
        pmax = phi(amax, n)
        cfg = self.cfg
        # running statistics (new slots start from their first observed value)
        fresh = self.age_steps == 0
        self.ema_fast = torch.where(fresh, p, cfg.ema_fast * self.ema_fast + (1 - cfg.ema_fast) * p)
        self.ema_slow = torch.where(fresh, p, cfg.ema_slow * self.ema_slow + (1 - cfg.ema_slow) * p)
        self.sum_phi = self.sum_phi + p
        self.max_phi = torch.maximum(self.max_phi, p)
        self.age_steps = self.age_steps + 1
        # hits: top-x% of visible slots by attention this chunk
        k = max(1, math.ceil(cfg.hit_top_frac * n))
        thresh = torch.topk(a, k).values.min()
        hit = a >= thresh
        self.hits = self.hits + hit.float()
        self.last_hit = torch.where(hit, torch.full_like(self.last_hit, st.step), self.last_hit)
        # per-chunk mass bookkeeping
        cids = st.chunk_id.to(dev)
        uniq, inv = torch.unique(cids, return_inverse=True)
        mass_per_chunk = torch.zeros(uniq.numel(), device=dev).index_add_(0, inv, a)
        counts = torch.zeros(uniq.numel(), device=dev).index_add_(0, inv, torch.ones_like(a))
        for i, cid in enumerate(uniq.tolist()):
            self.chunk_mass[cid] = self.chunk_mass.get(cid, 0.0) + float(mass_per_chunk[i])
            if cid not in self.chunk_size0:
                self.chunk_size0[cid] = int(counts[i])
                self.chunk_alive[cid] = int(counts[i])
        chunk_mass_vec = torch.tensor([self.chunk_mass[c] for c in uniq.tolist()], device=dev)[inv]
        chunk_size0_vec = torch.tensor(
            [self.chunk_size0[c] for c in uniq.tolist()], device=dev, dtype=torch.float
        )[inv]
        chunk_alive_vec = torch.tensor(
            [self.chunk_alive[c] for c in uniq.tolist()], device=dev, dtype=torch.float
        )[inv]
        self.sum_a = self.sum_a + a
        chunk_share = phi(self.sum_a / (chunk_mass_vec + 1e-9), 1.0)
        # ---- assemble token features
        pos = st.positions.to(dev).float()
        age = (st.step - cids).float().clamp_min(0)
        ctx = max(1, st.ctx_len)
        tok = torch.stack(
            [
                torch.log1p(age) / _LOG_512,
                pos / ctx,
                torch.log1p(pos) / _LOG_NMAX,
                st.is_generated.to(dev).float(),
                p,
                self.ema_fast,
                self.ema_slow,
                self.sum_phi / self.age_steps.clamp_min(1),
                self.max_phi,
                pmax,
                (torch.log((amax + 1e-6) / (a + 1e-6)) / 5.0).clamp(-1, 1),
                self.hits / self.age_steps.clamp_min(1),
                torch.where(
                    self.last_hit >= 0,
                    torch.log1p((st.step - self.last_hit).float().clamp_min(0)) / _LOG_512,
                    torch.ones_like(p),
                ),
                chunk_share,
                (st.k_norm.to(dev).float() - cfg.k_norm_mean) / cfg.k_norm_std,
                (st.v_norm.to(dev).float() - cfg.v_norm_mean) / cfg.v_norm_std,
                st.adj_cos.to(dev).float(),
                1.0 - chunk_alive_vec / chunk_size0_vec.clamp_min(1),
            ],
            dim=1,
        )
        tok = tok.index_select(1, self._sel)
        # ---- globals
        n_prot = int(st.protected_mask().sum())
        n_cand = max(1, n - n_prot)
        m = max(0, n - budget)
        total_steps = n_decisions_total or max(
            1, (st.n_prompt + st.max_new_tokens) // max(1, st.n_new)
        )
        ent = -(a[a > 0] * torch.log(a[a > 0])).sum() / math.log(max(2, n))
        glob = torch.tensor(
            [
                budget / max(1, st.n_prompt),
                n / max(1, budget),
                m / n_cand,
                math.log1p(st.ctx_len) / _LOG_NMAX,
                min(1.0, st.step / total_steps),
                float(st.phase),
                (st.max_new_tokens - st.n_generated) / max(1, st.max_new_tokens),
                float(ent),
            ],
            device=dev,
            dtype=torch.float32,
        )
        self.last_tok, self.last_glob = tok, glob
        return tok, glob

    @property
    def n_token_features(self) -> int:
        return len(self.cfg.token_features)
