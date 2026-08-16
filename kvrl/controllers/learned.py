"""Learned controllers: the PPO policy, the supervised regressor baseline, the sim oracle."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from kvrl.cache.view import CacheState
from kvrl.features import FeatureConfig, FeatureState
from kvrl.rl.policy import ScorePolicy, pad_batch
from kvrl.rl.sampler import deterministic_evict, log_prob, sample_evict

from .base import KVCacheController, select_keep


def save_policy_checkpoint(
    path: str | Path,
    policy: ScorePolicy,
    feature_cfg: FeatureConfig,
    kind: str = "rl",
    meta: dict | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": kind,
            "policy_state": policy.state_dict(),
            "policy_kwargs": {
                "n_tok": policy.n_tok,
                "n_glob": policy.n_glob,
                "arch": policy.arch,
                "hidden": policy.net[0].out_features if policy.arch == "mlp" else 128,
            },
            "feature_cfg": asdict(feature_cfg),
            "meta": meta or {},
        },
        p,
    )
    return p


def load_policy_checkpoint(path: str | Path) -> tuple[ScorePolicy, FeatureConfig, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    policy = ScorePolicy(**ck["policy_kwargs"])
    policy.load_state_dict(ck["policy_state"])
    policy.eval()
    return policy, FeatureConfig(**ck["feature_cfg"]), ck


class RLController(KVCacheController):
    """The trained eviction policy running on real inference (or in the simulator).

    Keeps its own :class:`FeatureState`, updated on every decision step via :meth:`observe`
    (identical code path to the simulator), scores every slot, and evicts ``m = n - budget``
    tokens (deterministic top-m by default; stochastic sampling optional).
    """

    name = "rl"
    needs_attention = True

    def __init__(
        self,
        policy: ScorePolicy,
        feature_cfg: FeatureConfig | None = None,
        deterministic: bool = True,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.policy = policy.to(device).eval()
        self.feature_cfg = feature_cfg or FeatureConfig()
        self.deterministic = deterministic
        self.device = device
        self.gen = torch.Generator().manual_seed(seed)
        self.fs = FeatureState(self.feature_cfg)
        self._obs = None
        self._chunk_ids = None
        self.last_scores: torch.Tensor | None = None
        self.last_logp: torch.Tensor | None = None

    def reset(self, **episode_info) -> None:
        self.fs = FeatureState(self.feature_cfg)
        self._obs = None
        self.last_scores = None

    def observe(self, state: CacheState, budget: int) -> None:
        tok, glob = self.fs.update(state, budget)
        self._obs = (tok, glob)
        self._chunk_ids = state.chunk_id

    @torch.no_grad()
    def decide(self, state: CacheState, budget: int) -> torch.Tensor:
        if self._obs is None or self._obs[0].shape[0] != state.n:
            self.observe(state, budget)
        tok, glob = self._obs
        n = state.n
        m = n - budget
        if m <= 0:
            return torch.arange(n)
        cand = ~state.protected_mask()
        t, g, valid, c, _ = pad_batch([tok], [glob], [cand], device=self.device)
        scores = self.policy(t, g, valid)[0]
        self.last_scores = scores.cpu()
        if self.deterministic:
            ev = deterministic_evict(scores, c[0], m)
        else:
            ev = sample_evict(scores, c[0], m, generator=self.gen)
        lp, _ = log_prob(scores, c[0], ev)
        self.last_logp = lp[0].cpu()
        keep = torch.ones(n, dtype=torch.bool)
        keep[ev.cpu()] = False
        return torch.nonzero(keep).flatten()

    def on_compact(self, keep_slots: torch.Tensor, n_before: int) -> None:
        self.fs.compact(keep_slots, self._chunk_ids)
        self._obs = None

    def importance(self, state: CacheState) -> torch.Tensor:
        """Per-slot importance for visualisation: -score (higher = more worth keeping)."""
        if self.last_scores is None or self.last_scores.numel() != state.n:
            self.decide(state, budget=state.n - 1)
        return -self.last_scores

    def confidence(self) -> torch.Tensor | None:
        """Per-slot keep-probability proxy: 1 - softmax(score) mass (for the dashboard)."""
        if self.last_scores is None:
            return None
        return 1.0 - torch.softmax(self.last_scores, 0)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "arch": self.policy.arch,
            "params": self.policy.n_params(),
            "deterministic": self.deterministic,
        }


class RegressorController(RLController):
    """Supervised baseline: same features, network trained to predict discounted future
    attention mass; evicts the lowest predictions (top-k on -prediction). Since the network
    outputs an eviction score = -predicted future mass, it plugs into the same machinery."""

    name = "regressor"


class OracleController(KVCacheController):
    """Sim-only upper reference: evict the tokens with the least discounted *future* mass."""

    name = "oracle"
    needs_attention = True

    def __init__(self, future_mass_fn):
        self.future_mass_fn = future_mass_fn

    def decide(self, state: CacheState, budget: int) -> torch.Tensor:
        fut = self.future_mass_fn()
        return select_keep(fut.float(), state, budget)


def make_learned_controller(
    name: str, checkpoint: str | Path | None = None, **kw
) -> KVCacheController:
    if checkpoint is None:
        raise ValueError(f"controller {name!r} needs checkpoint=<path>")
    policy, fcfg, ck = load_policy_checkpoint(checkpoint)
    cls = (
        RegressorController
        if name == "regressor" or ck.get("kind") == "regressor"
        else RLController
    )
    return cls(policy, fcfg, **kw)
