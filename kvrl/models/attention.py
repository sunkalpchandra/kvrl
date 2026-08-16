"""The ``kvrl`` attention implementation registered with transformers' AttentionInterface.

Two paths, chosen by the active :class:`AttentionContext`:

* **fast** – ``F.scaled_dot_product_attention`` with our own lower-right causal mask
  (key slot ``j`` is visible to chunk query ``i`` iff ``j <= kv - q + i``). This is correct
  for one-shot prefill (kv == q), decode (q == 1), a chunk appended after an arbitrarily
  *compacted* cache, and per-layer ragged caches (kv differs per layer, mask built per call).
* **stats** ("dual") – identical SDPA output plus a blocked softmax pass that accumulates the
  attention mass each cache slot received (summed over heads and the chunk's queries) into a
  :class:`kvrl.cache.stats.StatsBuffer`. Query sub-blocks bound the transient to
  ``H × qblock × kv`` floats — never the O(n²) map. Turning stats on cannot change outputs.

If a 4-D ``attention_mask`` (bool or additive float) is supplied it is used verbatim; that
is how the masked-reference oracle expresses eviction without touching the tensors.

Verified against transformers 5.15 (see .claude/context/design_inference_engineer.md):
signature ``(module, query[B,H,q,d], key/value[B,Hkv,kv,d], attention_mask, dropout,
scaling, **kwargs)`` → ``(out[B,q,H,d], None)``; ``module.layer_idx`` is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from kvrl.cache.stats import StatsBuffer

ATTN_NAME = "kvrl"


@dataclass
class AttentionContext:
    """Process-wide switches read by :func:`kvrl_attention` on every call."""

    stats: StatsBuffer | None = None
    stats_enabled: bool = False
    qblock: int = 64
    # per-layer 4-D masks for the ragged masked reference (layer_idx -> mask); None = unused
    layer_masks: dict[int, torch.Tensor] = field(default_factory=dict)
    check_finite: bool = False
    calls: int = 0


CTX = AttentionContext()


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, Hkv, kv, d] -> [B, Hkv*n_rep, kv, d] (grouped-query attention expansion)."""
    if n_rep == 1:
        return x
    b, hkv, s, d = x.shape
    return x[:, :, None, :, :].expand(b, hkv, n_rep, s, d).reshape(b, hkv * n_rep, s, d)


def causal_mask_lower_right(q: int, kv: int, device: torch.device) -> torch.Tensor | None:
    """Bool mask [1,1,q,kv]; True = attend. None when no masking is needed (q == 1)."""
    if q == 1:
        return None
    if kv < q:
        raise ValueError(f"kv ({kv}) < q ({q}): cache shorter than the chunk being appended")
    qi = torch.arange(q, device=device)[:, None]
    kj = torch.arange(kv, device=device)[None, :]
    return (kj <= (kv - q) + qi)[None, None]


def _accumulate_stats(query, key_rep, mask, scaling, layer_idx: int, ctx: AttentionContext):
    """Blocked softmax over query sub-blocks; adds per-slot mass to ctx.stats[layer_idx]."""
    assert ctx.stats is not None
    b, _, q, _ = query.shape
    kv = key_rep.shape[2]
    if b != 1:
        raise NotImplementedError("stats capture supports batch size 1")
    acc = torch.zeros(kv, dtype=torch.float32, device=query.device)
    kt = key_rep.transpose(2, 3)
    for s in range(0, q, ctx.qblock):
        e = min(s + ctx.qblock, q)
        scores = torch.matmul(query[:, :, s:e], kt) * scaling  # [1,H,qb,kv]
        if mask is not None:
            m = mask[:, :, s:e] if mask.shape[2] > 1 else mask
            if m.dtype == torch.bool:
                scores = scores.masked_fill(~m, float("-inf"))
            else:
                scores = scores + m
        p = torch.softmax(scores, dim=-1, dtype=torch.float32)
        acc += p.sum(dim=(0, 1, 2))
    ctx.stats.add(layer_idx, acc)


def kvrl_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
):
    ctx = CTX
    ctx.calls += 1
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    h, q = query.shape[1], query.shape[2]
    hkv, kv = key.shape[1], key.shape[2]
    n_rep = h // hkv
    layer_idx = getattr(module, "layer_idx", 0)

    if attention_mask is not None and attention_mask.dim() == 4:
        mask = attention_mask
    elif layer_idx in ctx.layer_masks:
        mask = ctx.layer_masks[layer_idx]
    else:
        mask = causal_mask_lower_right(q, kv, query.device)
    if mask is not None and mask.dtype != torch.bool and mask.dtype != query.dtype:
        mask = mask.to(query.dtype)

    key_rep = repeat_kv(key, n_rep)
    value_rep = repeat_kv(value, n_rep)
    out = F.scaled_dot_product_attention(
        query, key_rep, value_rep, attn_mask=mask, dropout_p=0.0, scale=scaling
    )
    if ctx.stats_enabled and ctx.stats is not None:
        _accumulate_stats(query, key_rep, mask, scaling, layer_idx, ctx)
        if layer_idx == 0:
            ctx.stats.note_queries(q)
    if ctx.check_finite and not torch.isfinite(out).all():
        raise FloatingPointError(f"non-finite attention output at layer {layer_idx}")
    return out.transpose(1, 2).contiguous(), None


_REGISTERED = False


def register_kvrl_attention() -> str:
    """Register once with transformers; returns the implementation name."""
    global _REGISTERED
    if not _REGISTERED:
        from transformers import AttentionInterface

        AttentionInterface.register(ATTN_NAME, kvrl_attention)
        _REGISTERED = True
    return ATTN_NAME
