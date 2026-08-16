"""Physical KV-cache eviction: keep a subset of slots in every layer via index_select."""

from __future__ import annotations

import torch


def cache_lengths(cache) -> list[int]:
    return [0 if layer.keys is None else int(layer.keys.shape[2]) for layer in cache.layers]


def compact_cache(cache, keep_slots: torch.Tensor) -> None:
    """Keep only ``keep_slots`` (sorted, unique LongTensor) in every layer (shared keep-set)."""
    for layer in cache.layers:
        if layer.keys is None:
            continue
        keep = keep_slots.to(layer.keys.device)
        layer.keys = layer.keys.index_select(2, keep)
        layer.values = layer.values.index_select(2, keep)


def compact_cache_per_layer(cache, keep_per_layer: list[torch.Tensor]) -> None:
    """Ragged eviction: a different keep-set per layer (v3 layer-specific policies)."""
    assert len(keep_per_layer) == len(cache.layers)
    for layer, keep_slots in zip(cache.layers, keep_per_layer):
        if layer.keys is None:
            continue
        keep = keep_slots.to(layer.keys.device)
        layer.keys = layer.keys.index_select(2, keep)
        layer.values = layer.values.index_select(2, keep)


def validate_keep(keep_slots: torch.Tensor, n: int) -> torch.Tensor:
    """Sorted unique LongTensor within [0, n). Raises on anything else."""
    if keep_slots.dtype != torch.long:
        keep_slots = keep_slots.long()
    if keep_slots.numel() == 0:
        raise ValueError("keep set must not be empty")
    if keep_slots.min() < 0 or keep_slots.max() >= n:
        raise IndexError(f"keep slots out of range [0,{n})")
    s = torch.unique(keep_slots, sorted=True)
    if s.numel() != keep_slots.numel():
        raise ValueError("keep slots must be unique")
    return s
