"""Plackett–Luce set actions via Gumbel-top-k, with exact per-slot log-probabilities.

Given per-token scores ``s`` (higher = more likely to be *evicted*) and a candidate mask,
sampling ``m`` tokens without replacement from softmax(s) is equivalent to taking the top-m
of ``s + Gumbel`` (Kool et al. 2019). For the ordered sample (i_1..i_m):

    p_j = s[i_j],  T_j = logsumexp_{l>=j} p_l,  Z_U = logsumexp over unpicked candidates,
    D_j = logaddexp(Z_U, T_j),  log π_j = p_j - D_j,   log π = Σ_j log π_j.

The Rao–Blackwellised entropy of the j-th conditional is
    H_j = D_j - e^{Z_U-D_j} w_U - e^{T_j-D_j} v_j,
    w_U = Σ_{l∈U} softmax_U(s)_l s_l,   v_j = Σ_{l>=j} e^{p_l-T_j} p_l.

Everything is batched over decisions with padding (scores padded with -inf, evict indices
padded with -1); shapes ``[B, N]`` and ``[B, M]``.
"""

from __future__ import annotations

import torch

NEG = float("-inf")


def _masked_scores(scores: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
    return scores.masked_fill(~cand_mask, NEG)


def sample_evict(
    scores: torch.Tensor,
    cand_mask: torch.Tensor,
    m: torch.Tensor | int,
    generator: torch.Generator | None = None,
    deterministic: bool = False,
):
    """Sample evict sets. scores/cand_mask [B, N]; m [B] (or int). Returns idx [B, M] (-1 pad)."""
    if scores.dim() == 1:
        scores, cand_mask = scores[None], cand_mask[None]
        squeeze = True
    else:
        squeeze = False
    B, N = scores.shape
    m_t = (
        torch.as_tensor(m, device=scores.device).expand(B)
        if not torch.is_tensor(m) or m.dim() == 0
        else m
    )
    M = int(m_t.max())
    s = _masked_scores(scores.float(), cand_mask)
    if deterministic:
        g = s
    else:
        u = torch.rand(s.shape, generator=generator, device=s.device).clamp_(1e-10, 1 - 1e-10)
        g = s - torch.log(-torch.log(u))
        g = g.masked_fill(~cand_mask, NEG)
    M = max(M, 1)
    top = torch.topk(g, k=min(M, N), dim=1).indices  # [B, M]
    slot = torch.arange(top.shape[1], device=s.device)[None, :]
    idx = torch.where(slot < m_t[:, None], top, torch.full_like(top, -1))
    if squeeze:
        idx = idx[0]
    return idx


def _picked_mask(s: torch.Tensor, idx: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Bool [B, N]: True where a valid evict slot points (duplicate/pad safe)."""
    counts = torch.zeros(s.shape, dtype=torch.long, device=s.device)
    counts.scatter_add_(1, idx, valid.long())
    return counts > 0


def log_prob(scores: torch.Tensor, cand_mask: torch.Tensor, evict_idx: torch.Tensor):
    """Per-slot log π_j for the ordered evict sets. Returns (logp [B, M], slot_mask [B, M])."""
    if scores.dim() == 1:
        scores, cand_mask, evict_idx = scores[None], cand_mask[None], evict_idx[None]
    s = _masked_scores(scores.float(), cand_mask)  # [B, N]
    valid = evict_idx >= 0  # [B, M]
    idx = evict_idx.clamp_min(0)
    p = torch.gather(s, 1, idx)  # [B, M]
    p = p.masked_fill(~valid, NEG)
    # unpicked partition
    s_u = s.masked_fill(_picked_mask(s, idx, valid), NEG)
    z_u = torch.logsumexp(s_u, dim=1)  # [B]
    # reverse cumulative logsumexp over picked scores (padding -inf at the tail is harmless)
    t = torch.flip(torch.logcumsumexp(torch.flip(p, [1]), dim=1), [1])  # [B, M]
    d = torch.logaddexp(z_u[:, None], t)
    logp = (p - d).masked_fill(~valid, 0.0)
    return logp, valid


def entropy(scores: torch.Tensor, cand_mask: torch.Tensor, evict_idx: torch.Tensor):
    """Rao–Blackwellised per-slot conditional entropies H_j (masked with slot_mask)."""
    if scores.dim() == 1:
        scores, cand_mask, evict_idx = scores[None], cand_mask[None], evict_idx[None]
    s = _masked_scores(scores.float(), cand_mask)
    valid = evict_idx >= 0
    idx = evict_idx.clamp_min(0)
    p = torch.gather(s, 1, idx).masked_fill(~valid, NEG)
    s_u = s.masked_fill(_picked_mask(s, idx, valid), NEG)
    z_u = torch.logsumexp(s_u, dim=1)  # [B]
    # w_U = Σ_{l∈U} softmax_U(s)_l s_l  (0 * -inf guarded)
    su_safe = s_u.masked_fill(torch.isinf(s_u), 0.0)
    w_u = (torch.exp(s_u - z_u[:, None]) * su_safe).sum(dim=1)
    # T_j = logsumexp_{l>=j} p_l ; v_j = Σ_{l>=j} e^{p_l - T_j} p_l (row-max shift for stability)
    t = torch.flip(torch.logcumsumexp(torch.flip(p, [1]), dim=1), [1])
    c = p.masked_fill(~valid, NEG).max(dim=1, keepdim=True).values  # [B,1]
    e = torch.exp(p - c) * p.masked_fill(~valid, 0.0)  # e^{p_l - c} p_l, 0 on padding
    tail = torch.flip(torch.cumsum(torch.flip(e, [1]), dim=1), [1])  # Σ_{l>=j} e^{p_l-c} p_l
    v = tail * torch.exp(c - t)
    d = torch.logaddexp(z_u[:, None], t)
    h = d - torch.exp(z_u[:, None] - d) * w_u[:, None] - torch.exp(t - d) * v
    return h.masked_fill(~valid, 0.0), valid


def deterministic_evict(scores: torch.Tensor, cand_mask: torch.Tensor, m: int) -> torch.Tensor:
    """Top-m by score (evaluation policy)."""
    s = _masked_scores(scores.float(), cand_mask)
    return torch.topk(s, k=m).indices


def exact_set_log_prob_bruteforce(
    scores: torch.Tensor, cand_mask: torch.Tensor, evict_set: set[int]
) -> float:
    """Reference for tests: log P(set) summing over all orderings (tiny n only)."""
    import itertools
    import math

    s = _masked_scores(scores.float(), cand_mask)
    items = sorted(evict_set)
    total = 0.0
    for perm in itertools.permutations(items):
        idx = torch.tensor(perm)
        lp, _ = log_prob(s, cand_mask, idx)
        total += math.exp(lp.sum().item())
    return math.log(total)
