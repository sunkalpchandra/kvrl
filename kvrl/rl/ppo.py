"""PPO for set-valued eviction actions (per-slot clipped surrogate, shared advantage)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .policy import ScorePolicy, ValueNet, pad_batch
from .sampler import entropy, log_prob


@dataclass
class Transition:
    tok: torch.Tensor  # [n, F] fp16
    glob: torch.Tensor  # [G]
    cand: torch.Tensor  # bool [n]
    evict: torch.Tensor  # [m] long
    logp_old: torch.Tensor  # [m]
    value: float
    reward: float
    priv: torch.Tensor  # [P]
    episode: int


@dataclass
class RolloutBuffer:
    gamma: float = 0.99
    lam: float = 0.95
    items: list[Transition] = field(default_factory=list)
    episode_ends: list[int] = field(default_factory=list)  # index after the last item of each ep
    last_values: list[float] = field(
        default_factory=list
    )  # bootstrap value per episode (0 if terminal)

    def add(self, tr: Transition) -> None:
        self.items.append(tr)

    def end_episode(self, last_value: float = 0.0) -> None:
        self.episode_ends.append(len(self.items))
        self.last_values.append(last_value)

    def __len__(self) -> int:
        return len(self.items)

    def compute_gae(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (advantages, returns) aligned with items (episode-ordered buffer)."""
        n = len(self.items)
        adv = np.zeros(n, dtype=np.float32)
        ret = np.zeros(n, dtype=np.float32)
        start = 0
        ends = list(self.episode_ends)
        if not ends or ends[-1] != n:  # partial last episode: bootstrap with 0 (conservative)
            ends.append(n)
            self.last_values.append(0.0)
        for end, last_v in zip(ends, self.last_values):
            gae = 0.0
            next_v = last_v
            for i in range(end - 1, start - 1, -1):
                t = self.items[i]
                delta = t.reward + self.gamma * next_v - t.value
                gae = delta + self.gamma * self.lam * gae
                adv[i] = gae
                ret[i] = gae + t.value
                next_v = t.value
            start = end
        return torch.from_numpy(adv), torch.from_numpy(ret)

    def clear(self) -> None:
        self.items.clear()
        self.episode_ends.clear()
        self.last_values.clear()


@dataclass
class PPOConfig:
    lr: float = 3e-4
    lr_end: float = 3e-5
    clip: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch: int = 256
    target_kl: float = 0.02
    normalize_adv: bool = True
    ratio_mode: str = "per_slot"  # or "sequence"


class PPO:
    def __init__(self, policy: ScorePolicy, value: ValueNet, cfg: PPOConfig, device="cpu"):
        self.policy = policy.to(device)
        self.value = value.to(device)
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters()), lr=cfg.lr, eps=1e-5
        )

    def set_lr(self, frac_done: float) -> float:
        lr = self.cfg.lr + (self.cfg.lr_end - self.cfg.lr) * min(1.0, max(0.0, frac_done))
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

    @torch.no_grad()
    def act(self, tok, glob, cand, m: int, priv=None, generator=None, deterministic=False):
        """One decision. Returns evict idx [m], per-slot logp [m], value (float), scores [n]."""
        from .sampler import deterministic_evict, sample_evict

        t, g, valid, c, _ = pad_batch([tok], [glob], [cand], device=self.device)
        scores = self.policy(t, g, valid)[0]
        cmask = c[0]
        if deterministic:
            ev = deterministic_evict(scores, cmask, m)
        else:
            ev = sample_evict(scores, cmask, m, generator=generator)
        lp, _ = log_prob(scores, cmask, ev)
        pv = None if priv is None else priv[None].to(self.device)
        v = self.value(t, g, valid, pv)[0].item()
        return ev.cpu(), lp[0].cpu(), v, scores.cpu()

    def update(self, buf: RolloutBuffer) -> dict:
        cfg = self.cfg
        adv_all, ret_all = buf.compute_gae()
        if cfg.normalize_adv and adv_all.numel() > 1:
            adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)
        n = len(buf)
        idx_all = np.arange(n)
        stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clipfrac": [],
            "ratio_max": [],
        }
        stop = False
        epochs_done = 0
        for _epoch in range(cfg.epochs):
            epochs_done += 1
            np.random.shuffle(idx_all)
            for s in range(0, n, cfg.minibatch):
                mb = idx_all[s : s + cfg.minibatch]
                items = [buf.items[i] for i in mb]
                tok, glob, valid, cand, ev = pad_batch(
                    [t.tok for t in items],
                    [t.glob for t in items],
                    [t.cand for t in items],
                    [t.evict for t in items],
                    device=self.device,
                )
                priv = torch.stack([t.priv for t in items]).to(self.device)
                old_lp = torch.zeros_like(ev, dtype=torch.float32)
                for i, t in enumerate(items):
                    old_lp[i, : t.logp_old.numel()] = t.logp_old.to(self.device)
                old_v = torch.tensor([t.value for t in items], device=self.device)
                adv = adv_all[mb].to(self.device)
                ret = ret_all[mb].to(self.device)
                scores = self.policy(tok, glob, valid)
                lp, slot_mask = log_prob(scores, cand, ev)
                sm = slot_mask.float()
                n_slots = sm.sum(dim=1).clamp_min(1.0)
                if cfg.ratio_mode == "sequence":
                    ratio = torch.exp(((lp - old_lp) * sm).sum(dim=1))[:, None].expand_as(lp)
                else:
                    ratio = torch.exp(lp - old_lp)
                a = adv[:, None]
                surr1 = ratio * a
                surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a
                pl = -(torch.minimum(surr1, surr2) * sm).sum(dim=1) / n_slots
                policy_loss = pl.mean()
                ent, _ = entropy(scores, cand, ev)
                ent_mean = ((ent * sm).sum(dim=1) / n_slots).mean()
                v = self.value(tok, glob, valid, priv)
                v_clipped = old_v + torch.clamp(v - old_v, -cfg.value_clip, cfg.value_clip)
                value_loss = torch.maximum((v - ret) ** 2, (v_clipped - ret) ** 2).mean() * 0.5
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * ent_mean
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    cfg.max_grad_norm,
                )
                self.opt.step()
                with torch.no_grad():
                    log_ratio = (lp - old_lp) * sm
                    approx_kl = (((torch.exp(log_ratio) - 1) - log_ratio) * sm).sum() / sm.sum()
                    clipfrac = (((ratio - 1).abs() > cfg.clip).float() * sm).sum() / sm.sum()
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(ent_mean.item())
                stats["approx_kl"].append(approx_kl.item())
                stats["clipfrac"].append(clipfrac.item())
                stats["ratio_max"].append(ratio[slot_mask].max().item())
                if approx_kl.item() > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        out = {k: float(np.mean(v)) for k, v in stats.items() if v}
        out["epochs_done"] = epochs_done
        out["early_stop"] = stop
        # explained variance of the value function
        with torch.no_grad():
            vals = torch.tensor([t.value for t in buf.items])
            var = ret_all.var()
            out["explained_variance"] = float(1 - (ret_all - vals).var() / (var + 1e-8))
        return out
