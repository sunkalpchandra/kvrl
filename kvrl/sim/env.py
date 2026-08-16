"""CacheSimEnv — replay a recorded full-cache trace under an eviction policy.

The simulator mirrors :class:`kvrl.engine.InferenceEngine` step for step: after chunk k
enters the cache the controller must evict ``m = n - B`` tokens (hard budget). Attention
statistics come from the trace and are renormalised over the retained set (what a real
evicted-cache model would approximately see); the reward charges each eviction its
discounted *future* full-cache attention mass (R1, see ML_SPEC.md):

    r_k = -(1 / r_scale) * Σ_{j ∈ E_k} F^γ_k(j),   F^γ_k(j) = Σ_{k'>k} γ^{k'-k} A_{k'}(j)

plus an optional terminal task term (fraction of task-critical tokens retained through
generation). Everything a controller sees is a :class:`CacheState`, so heuristics run
unchanged here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from kvrl.cache.view import CacheState
from kvrl.features import FeatureConfig, FeatureState
from kvrl.traces.storage import Trace


def episode_budget(budget_frac: float, n_prompt: int, chunk: int, min_tokens: int = 128) -> int:
    if budget_frac >= 1.0:
        return 1 << 30
    b = int(math.ceil(budget_frac * n_prompt / chunk) * chunk)
    return max(min_tokens, b)


@dataclass
class SimStepResult:
    obs_tok: torch.Tensor  # [n, F] features for the *next* decision (or empty if done)
    obs_glob: torch.Tensor  # [G]
    cand_mask: torch.Tensor  # bool [n] candidates for eviction (False = protected)
    m: int  # number of tokens that must be evicted at the next decision (0 = no decision)
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class CacheSimEnv:
    def __init__(
        self,
        gamma: float = 0.99,
        r_scale: float = 1.0,
        lambda_task: float = 1.0,
        n_sink: int = 4,
        feature_cfg: FeatureConfig | None = None,
        min_budget_tokens: int = 128,
        use_layer_max_reward: bool = False,
    ):
        self.gamma = gamma
        self.r_scale = r_scale
        self.lambda_task = lambda_task
        self.n_sink = n_sink
        self.feature_cfg = feature_cfg or FeatureConfig()
        self.min_budget_tokens = min_budget_tokens
        self.use_layer_max_reward = use_layer_max_reward
        self.trace: Trace | None = None

    # ------------------------------------------------------------------ setup
    def _precompute(self, tr: Trace) -> None:
        A = tr.attn_mean.astype(np.float32)  # [K, T]
        self.A = A
        self.Amax = tr.attn_lmax.astype(np.float32)
        K, T = A.shape
        src = self.Amax if self.use_layer_max_reward else A
        # discounted future mass F[k, j] = Σ_{k'>k} γ^{k'-k} A_{k'}(j)  (reverse scan)
        F = np.zeros_like(src)
        acc = np.zeros(T, dtype=np.float32)
        for k in range(K - 1, -1, -1):
            F[k] = acc
            acc = src[k] + self.gamma * acc
        self.F = F
        # chunk id per token from step_end
        ends = tr.step_end
        tok_chunk = np.zeros(T, dtype=np.int64)
        start = 0
        for k, e in enumerate(ends):
            tok_chunk[start:e] = k
            start = e
        self.tok_chunk = tok_chunk
        self.T = T
        self.K = K

    def reset(
        self,
        trace: Trace,
        budget_frac: float | None = None,
        budget: int | None = None,
        max_new_tokens: int | None = None,
    ) -> SimStepResult:
        self.trace = trace
        self._precompute(trace)
        if budget is None:
            assert budget_frac is not None
            budget = episode_budget(
                budget_frac, trace.n_prompt, trace.chunk, self.min_budget_tokens
            )
        self.budget = int(budget)
        self.max_new_tokens = max_new_tokens or trace.n_gen or 1
        self.k = 0  # next step to process
        self.alive = np.zeros(0, dtype=np.int64)  # retained token indices (sorted)
        self.evicted = np.zeros(self.T, dtype=bool)
        self.evict_step = np.full(self.T, -1, dtype=np.int64)
        self.fs = FeatureState(self.feature_cfg)
        self.total_reward = 0.0
        self.lost_mass = []  # ℓ_k per step (raw full-cache mass on already-evicted tokens)
        self.crit_retained = []  # per decode step
        self.n_evictions = 0
        self._cum_mean = np.zeros(0, dtype=np.float32)
        self._cum_max = np.zeros(0, dtype=np.float32)
        self._pending_state: CacheState | None = None
        return self._advance(reward=0.0)

    # ------------------------------------------------------------------ core
    def _build_state(self, k: int) -> CacheState:
        tr = self.trace
        assert tr is not None
        end = int(tr.step_end[k])
        start = int(tr.step_end[k - 1]) if k > 0 else 0
        new = np.arange(start, end)
        R = np.concatenate([self.alive, new])
        A_row = self.A[k, :end]
        # lost mass on this chunk (tokens evicted before step k)
        ev = self.evicted[:end]
        self.lost_mass.append(float(A_row[ev].sum()))
        a_R = A_row[R]
        z = float(a_R.sum())
        if z <= 0:
            a_hat = np.full(R.shape[0], 1.0 / max(1, R.shape[0]), dtype=np.float32)
            amax_hat = a_hat.copy()
        else:
            a_hat = a_R / z
            amax_hat = self.Amax[k, :end][R] / z
        phase = int(tr.step_phase[k])
        n_generated = max(0, end - tr.n_prompt)
        st = CacheState(
            positions=torch.from_numpy(R),
            chunk_id=torch.from_numpy(self.tok_chunk[R]),
            is_generated=torch.from_numpy(R >= tr.n_prompt),
            attn_last_mean=torch.from_numpy(a_hat.astype(np.float32)),
            attn_last_max=torch.from_numpy(amax_hat.astype(np.float32)),
            attn_cum_mean=torch.zeros(R.shape[0]),  # filled below from FeatureState-free sums
            attn_cum_max=torch.zeros(R.shape[0]),
            k_norm=torch.from_numpy(tr.key_norm[R].astype(np.float32)),
            v_norm=torch.from_numpy(tr.value_norm[R].astype(np.float32)),
            adj_cos=torch.from_numpy(tr.adj_key_cos[R].astype(np.float32)),
            n_new=int(new.shape[0]),
            step=k,
            ctx_len=end,
            phase=phase,
            n_prompt=tr.n_prompt,
            n_generated=n_generated,
            max_new_tokens=self.max_new_tokens,
            n_sink=self.n_sink,
        )
        self.alive = R
        return st

    def _advance(self, reward: float) -> SimStepResult:
        """Move to the next decision point (skipping no-op steps), returning the observation."""
        while True:
            if self.k >= self.K:
                return self._finish(reward)
            st = self._build_state(self.k)
            # cumulative attention (H2O statistic) maintained here for heuristics
            q = st.n_new
            self._cum_mean = np.concatenate([self._cum_mean, np.zeros(q, dtype=np.float32)])
            self._cum_max = np.concatenate([self._cum_max, np.zeros(q, dtype=np.float32)])
            self._cum_mean += st.attn_last_mean.numpy()
            self._cum_max += st.attn_last_max.numpy()
            st.attn_cum_mean = torch.from_numpy(self._cum_mean.copy())
            st.attn_cum_max = torch.from_numpy(self._cum_max.copy())
            tok, glob = self.fs.update(st, self.budget)
            if st.phase == 1 and self.trace is not None and self.trace.critical_mask.any():
                crit = self.trace.critical_mask
                self.crit_retained.append(float(crit[self.alive].sum() / crit.sum()))
            n = st.n
            m = max(0, n - self.budget)
            if m == 0:
                # no decision: keep everything, continue
                self.fs.compact(torch.arange(n), st.chunk_id)  # no-op compaction keeps arrays
                self.k += 1
                self._last_state = st
                continue
            cand = ~st.protected_mask()
            self._pending_state = st
            self._pending = (tok, glob, cand, m)
            return SimStepResult(tok, glob, cand, m, reward, False, {"step": self.k, "n": n})

    def step(self, evict_slots: torch.Tensor | np.ndarray) -> SimStepResult:
        """Evict the given slots (indices into the current retained set) and advance."""
        assert self._pending_state is not None, "call reset() first / episode finished"
        st = self._pending_state
        _tok, _glob, cand, m = self._pending
        ev = torch.as_tensor(evict_slots).long().flatten()
        n = st.n
        if ev.numel() != m:
            raise ValueError(f"must evict exactly {m} slots, got {ev.numel()}")
        if ev.numel() and (ev.min() < 0 or ev.max() >= n):
            raise IndexError("evict slot out of range")
        if not cand[ev].all():
            raise ValueError("attempted to evict a protected slot")
        keep_mask = torch.ones(n, dtype=torch.bool)
        keep_mask[ev] = False
        keep = torch.nonzero(keep_mask).flatten()
        # reward: discounted future mass of evicted tokens
        k = self.k
        ev_tokens = self.alive[ev.numpy()]
        fut = float(self.F[k, ev_tokens].sum())
        reward = -fut / self.r_scale
        self.evicted[ev_tokens] = True
        self.evict_step[ev_tokens] = k
        self.n_evictions += int(ev.numel())
        self.fs.compact(keep, st.chunk_id)
        self.alive = self.alive[keep.numpy()]
        self._cum_mean = self._cum_mean[keep.numpy()]
        self._cum_max = self._cum_max[keep.numpy()]
        self.total_reward += reward
        self._pending_state = None
        self.k += 1
        return self._advance(reward)

    def _finish(self, reward: float) -> SimStepResult:
        tr = self.trace
        assert tr is not None
        info = self.metrics()
        term = 0.0
        if self.lambda_task > 0 and tr.critical_mask.any() and self.crit_retained:
            term = self.lambda_task * float(np.mean(self.crit_retained))
        self.total_reward += term
        info["terminal_task_reward"] = term
        info["total_reward"] = self.total_reward
        return SimStepResult(
            torch.zeros(0, self.fs.n_token_features),
            torch.zeros(8),
            torch.zeros(0, dtype=torch.bool),
            0,
            reward + term,
            True,
            info,
        )

    # ------------------------------------------------------------------ helpers
    def keep_to_evict(self, keep_slots: torch.Tensor) -> torch.Tensor:
        """Convert a controller's keep-set (slots) into the evict-set the env expects."""
        st = self._pending_state
        assert st is not None
        n = st.n
        mask = torch.ones(n, dtype=torch.bool)
        mask[keep_slots.long()] = False
        return torch.nonzero(mask).flatten()

    @property
    def state(self) -> CacheState | None:
        return self._pending_state

    def future_mass(self) -> torch.Tensor:
        """Privileged: F^γ_k(j) for the current retained set (oracle / critic only)."""
        assert self._pending_state is not None
        return torch.from_numpy(self.F[self.k, self.alive].astype(np.float32))

    def metrics(self) -> dict:
        tr = self.trace
        assert tr is not None
        lm = np.array(self.lost_mass, dtype=np.float64)
        dec = tr.step_phase == 1
        out = {
            "lost_mass_mean": float(lm.mean()) if lm.size else 0.0,
            "lost_mass_decode": float(lm[dec[: lm.size]].mean()) if dec.any() and lm.size else 0.0,
            "lost_mass_last_prefill": float(lm[(~dec)[: lm.size]][-1])
            if lm.size and (~dec).any()
            else 0.0,
            "n_evictions": self.n_evictions,
            "budget": self.budget,
            "final_cache": int(self.alive.shape[0]),
            "crit_retained": float(np.mean(self.crit_retained))
            if self.crit_retained
            else float("nan"),
        }
        return out


def run_controller_episode(
    env: CacheSimEnv, trace: Trace, controller, budget_frac: float, **reset_kw
) -> dict:
    """Roll one episode with any :class:`KVCacheController` (heuristics or learned)."""
    res = env.reset(trace, budget_frac=budget_frac, **reset_kw)
    controller.reset(
        episode=0, n_prompt=trace.n_prompt, budget=env.budget, max_new_tokens=env.max_new_tokens
    )
    while not res.done:
        st = env.state
        keep = controller.decide(st, env.budget)
        controller.on_compact(keep, st.n)
        res = env.step(env.keep_to_evict(keep))
    return res.info
