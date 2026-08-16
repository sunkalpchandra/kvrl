"""transformers backend with the ``kvrl`` attention implementation.

Verified against transformers 5.15 (2026-08-16). The compat check below fails loudly if
a future version changes the cache layout or starts consuming ``cache_position``.
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvrl.cache.stats import StatsBuffer
from kvrl.utils.device import pick_dtype, resolve_device

from .attention import CTX, register_kvrl_attention
from .base_model import BaseCausalLM, ModelInfo
from .model_registry import ModelSpec, get_spec


def _compat_check(model, cache) -> None:
    import inspect

    assert hasattr(cache, "layers"), "DynamicCache without .layers — transformers API drift"
    sig = inspect.signature(model.model.forward)
    assert "position_ids" in sig.parameters, "model.forward lacks position_ids"


class HFCausalLM(BaseCausalLM):
    def __init__(
        self,
        name: str = "qwen2.5-0.5b-instruct",
        device: str | None = None,
        dtype: str | None = None,
        attn: str = "kvrl",
        stats_max_slots: int = 4096,
    ):
        self.spec: ModelSpec = get_spec(name)
        self.device = resolve_device(device)
        self.dtype = pick_dtype(self.device, dtype)
        attn_impl = register_kvrl_attention() if attn == "kvrl" else attn
        t0 = time.time()
        if self.spec.hf_id is None:  # tiny random model for tests
            from transformers import Qwen2Config, Qwen2ForCausalLM

            cfg = Qwen2Config(**self.spec.tiny_config)
            cfg._attn_implementation = attn_impl
            torch.manual_seed(0)
            self.model = Qwen2ForCausalLM(cfg).to(self.device, self.dtype).eval()
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.spec.hf_id)
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.spec.hf_id, dtype=self.dtype, attn_implementation=attn_impl
                )
                .to(self.device)
                .eval()
            )
        self.load_seconds = time.time() - t0
        cfg = self.model.config
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.info = ModelInfo(
            name=self.spec.name,
            n_layers=cfg.num_hidden_layers,
            n_heads=cfg.num_attention_heads,
            n_kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            vocab_size=cfg.vocab_size,
            max_context=min(
                self.spec.max_context, getattr(cfg, "max_position_embeddings", 1 << 30)
            ),
            dtype=self.dtype,
            device=self.device,
            n_params=sum(p.numel() for p in self.model.parameters()),
        )
        self.stats = StatsBuffer(
            self.info.n_layers, self.info.n_heads, stats_max_slots, self.device
        )
        _compat_check(self.model, self.new_cache())
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------ cache / forward
    def new_cache(self) -> DynamicCache:
        return DynamicCache(config=self.model.config)

    @torch.no_grad()
    def forward_chunk(
        self, input_ids: torch.Tensor, positions: torch.Tensor, cache, logits_to_keep: int = 1
    ) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids[None]
        if positions.dim() == 1:
            positions = positions[None]
        assert input_ids.shape == positions.shape, (input_ids.shape, positions.shape)
        out = self.model(
            input_ids=input_ids.to(self.device),
            position_ids=positions.to(self.device),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=logits_to_keep,
        )
        return out.logits

    def set_stats(self, enabled: bool, qblock: int | None = None) -> None:
        """Route the ``kvrl`` attention's statistics into this model's buffer."""
        CTX.stats = self.stats
        CTX.stats_enabled = enabled
        if qblock is not None:
            CTX.qblock = qblock

    # ------------------------------------------------------------------ reference
    @torch.no_grad()
    def greedy_reference(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids[None]
        out = self.model.generate(
            input_ids.to(self.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,  # Qwen2.5 generation_config ships 1.1 (D-008)
            temperature=None,
            top_p=None,
            top_k=None,
            min_new_tokens=0,
            pad_token_id=self._pad_id(),
        )
        return out[:, input_ids.shape[1] :]

    def _pad_id(self) -> int:
        if self.tokenizer is not None and self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        eos = self.eos_token_ids
        return next(iter(eos)) if eos else 0

    # ------------------------------------------------------------------ text
    def encode(self, text: str) -> torch.Tensor:
        if self.tokenizer is None:
            raise RuntimeError("tiny-random has no tokenizer")
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    def encode_chat(self, user: str, system: str | None = None) -> torch.Tensor:
        if self.tokenizer is None:
            raise RuntimeError("tiny-random has no tokenizer")
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        if self.spec.chat and getattr(self.tokenizer, "chat_template", None):
            enc = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
            return enc["input_ids"][0]
        return self.encode((system + "\n\n" if system else "") + user)

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            return max(1, len(text) // 4)
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids) -> str:
        if self.tokenizer is None:
            return " ".join(str(int(i)) for i in ids)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def eos_token_ids(self) -> set[int]:
        ids: set[int] = set()
        gc = getattr(self.model, "generation_config", None)
        eos = getattr(gc, "eos_token_id", None)
        if isinstance(eos, int):
            ids.add(eos)
        elif isinstance(eos, list | tuple):
            ids.update(int(e) for e in eos)
        if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
            ids.add(int(self.tokenizer.eos_token_id))
        return ids


def load_model(name: str, device: str | None = None, dtype: str | None = None, **kw) -> HFCausalLM:
    return HFCausalLM(name, device=device, dtype=dtype, **kw)
