"""Policy and value networks over ragged token sets (padded batches).

Policy: per-token *eviction score* s_i (higher = more likely evicted). Three variants:
  mlp       — per-token MLP over [tok ⊕ glob] (v1, ~20K params, set-agnostic)
  deepsets  — per-token encoder + mean/max pooled context fed back to every token (v1.5)
  setattn   — 2 ISAB-style induced set-attention blocks (v2, ~75K params)
Value: per-token encoder → mean⊕max pool ⊕ glob ⊕ privileged (sim-only) → MLP → scalar.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(i: int, h: int, o: int, n_layers: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = i
    for _ in range(n_layers):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def masked_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mean ⊕ max over valid tokens. x [B,N,H], mask [B,N] -> [B,2H]"""
    m = mask.unsqueeze(-1).to(x.dtype)
    cnt = m.sum(dim=1).clamp_min(1.0)
    mean = (x * m).sum(dim=1) / cnt
    mx = x.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], dim=-1)


class ISAB(nn.Module):
    """Induced set attention block (Lee et al. 2019): O(N·k) instead of O(N²)."""

    def __init__(self, d: int, n_heads: int = 4, n_induce: int = 32, ffn: int = 128):
        super().__init__()
        self.induce = nn.Parameter(torch.randn(1, n_induce, d) * 0.02)
        self.mab1 = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.mab2 = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.SiLU(), nn.Linear(ffn, d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        ind = self.induce.expand(B, -1, -1)
        h, _ = self.mab1(ind, x, x, key_padding_mask=~mask, need_weights=False)  # [B,k,d]
        h = self.ln1(ind + h)
        y, _ = self.mab2(x, h, h, need_weights=False)  # [B,N,d]
        y = self.ln2(x + y)
        return y + self.ff(y)


class ScorePolicy(nn.Module):
    def __init__(
        self,
        n_tok: int,
        n_glob: int,
        hidden: int = 128,
        arch: str = "mlp",
        n_layers: int = 2,
        d_set: int = 64,
    ):
        super().__init__()
        self.arch = arch
        self.n_tok, self.n_glob = n_tok, n_glob
        if arch == "mlp":
            self.net = _mlp(n_tok + n_glob, hidden, 1, n_layers)
        elif arch == "deepsets":
            self.phi = _mlp(n_tok + n_glob, hidden, d_set, 1)
            self.ctx = nn.Sequential(nn.Linear(2 * d_set + n_glob, d_set), nn.SiLU())
            self.psi = _mlp(d_set + d_set, hidden, 1, 1)
        elif arch == "setattn":
            self.embed = nn.Linear(n_tok + n_glob, d_set)
            self.blocks = nn.ModuleList([ISAB(d_set) for _ in range(2)])
            self.head = nn.Linear(d_set, 1)
        else:
            raise ValueError(f"unknown policy arch {arch!r}")

    def forward(self, tok: torch.Tensor, glob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """tok [B,N,F], glob [B,G], mask [B,N] (valid tokens) -> scores [B,N]"""
        B, N, _ = tok.shape
        x = torch.cat([tok, glob[:, None, :].expand(B, N, -1)], dim=-1)
        if self.arch == "mlp":
            return self.net(x).squeeze(-1)
        if self.arch == "deepsets":
            h = F.silu(self.phi(x))
            ctx = self.ctx(torch.cat([masked_pool(h, mask), glob], dim=-1))  # [B,d]
            z = torch.cat([h, ctx[:, None, :].expand(B, N, -1)], dim=-1)
            return self.psi(z).squeeze(-1)
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(h).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ValueNet(nn.Module):
    def __init__(self, n_tok: int, n_glob: int, n_priv: int = 3, hidden: int = 128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_tok + n_glob, hidden), nn.SiLU())
        self.head = _mlp(2 * hidden + n_glob + n_priv, hidden, 1, 1)
        self.n_priv = n_priv

    def forward(
        self,
        tok: torch.Tensor,
        glob: torch.Tensor,
        mask: torch.Tensor,
        priv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, _ = tok.shape
        x = torch.cat([tok, glob[:, None, :].expand(B, N, -1)], dim=-1)
        h = self.enc(x)
        pooled = masked_pool(h, mask)
        if priv is None:
            priv = torch.zeros(B, self.n_priv, device=tok.device, dtype=tok.dtype)
        return self.head(torch.cat([pooled, glob, priv], dim=-1)).squeeze(-1)


def pad_batch(
    toks: list[torch.Tensor],
    globs: list[torch.Tensor],
    cands: list[torch.Tensor],
    evicts: list[torch.Tensor] | None = None,
    device="cpu",
):
    """Pad ragged decisions into [B,N,F], [B,G], valid [B,N], cand [B,N], evict [B,M]."""
    B = len(toks)
    N = max(t.shape[0] for t in toks)
    Fd = toks[0].shape[1]
    tok = torch.zeros(B, N, Fd, device=device)
    valid = torch.zeros(B, N, dtype=torch.bool, device=device)
    cand = torch.zeros(B, N, dtype=torch.bool, device=device)
    for i, (t, c) in enumerate(zip(toks, cands)):
        n = t.shape[0]
        tok[i, :n] = t.to(device, torch.float32)
        valid[i, :n] = True
        cand[i, :n] = c.to(device)
    glob = torch.stack([g.to(device, torch.float32) for g in globs])
    ev = None
    if evicts is not None:
        M = max(1, max(e.numel() for e in evicts))
        ev = torch.full((B, M), -1, dtype=torch.long, device=device)
        for i, e in enumerate(evicts):
            ev[i, : e.numel()] = e.to(device)
    return tok, glob, valid, cand, ev
