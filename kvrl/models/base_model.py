"""Abstract interface every backend must implement (kept small on purpose)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelInfo:
    name: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    max_context: int
    dtype: torch.dtype
    device: torch.device
    n_params: int

    @property
    def kv_bytes_per_token(self) -> int:
        return (
            self.n_layers * 2 * self.n_kv_heads * self.head_dim * torch.finfo(self.dtype).bits // 8
        )

    def kv_bytes(self, n_tokens: int) -> int:
        return self.kv_bytes_per_token * n_tokens


class BaseCausalLM(ABC):
    info: ModelInfo

    @abstractmethod
    def new_cache(self):
        """Fresh, empty KV cache object for this backend."""

    @abstractmethod
    def forward_chunk(
        self, input_ids: torch.Tensor, positions: torch.Tensor, cache, logits_to_keep: int = 1
    ) -> torch.Tensor:
        """Append ``input_ids`` [1, q] at absolute ``positions`` [q]; return logits [1, k, V]."""

    @abstractmethod
    def greedy_reference(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Backend-native greedy generation used only as a correctness oracle. Returns [1, G]."""

    @abstractmethod
    def encode(self, text: str) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, ids: torch.Tensor | list[int]) -> str: ...

    @property
    @abstractmethod
    def eos_token_ids(self) -> set[int]: ...
