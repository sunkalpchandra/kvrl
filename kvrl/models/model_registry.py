"""Model registry: short names -> Hugging Face ids + per-model defaults.

Nothing else in the codebase hardcodes a model id. ``tiny-random`` builds a 2-layer random
Qwen2 for CPU tests (no download).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str | None
    max_context: int
    chat: bool = True
    notes: str = ""
    # only for synthetic test models
    tiny_config: dict = field(default_factory=dict)


REGISTRY: dict[str, ModelSpec] = {
    "qwen2.5-0.5b-instruct": ModelSpec(
        "qwen2.5-0.5b-instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        32768,
        notes="24L, 14 Q heads, 2 KV heads, d64; 12,288 B KV/token fp16. Primary model.",
    ),
    "qwen2.5-1.5b-instruct": ModelSpec(
        "qwen2.5-1.5b-instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        32768,
        notes="28L, 12 Q heads, 2 KV heads, d128; ~3 GB fp16 — tight on 8 GB.",
    ),
    "smollm2-360m-instruct": ModelSpec(
        "smollm2-360m-instruct",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        8192,
        notes="32L, 15 Q heads, 5 KV heads, d64; 40,960 B KV/token fp16.",
    ),
    "qwen3-0.6b": ModelSpec(
        "qwen3-0.6b",
        "Qwen/Qwen3-0.6B",
        40960,
        notes="28L, 16 Q heads, 8 KV heads, d128; 114,688 B KV/token fp16 (KV-heavy).",
    ),
    "tiny-random": ModelSpec(
        "tiny-random",
        None,
        4096,
        chat=False,
        notes="2-layer random Qwen2 for tests (no download).",
        tiny_config={
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 1000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "tie_word_embeddings": True,
        },
    ),
}


def get_spec(name: str) -> ModelSpec:
    key = name.lower()
    if key in REGISTRY:
        return REGISTRY[key]
    # allow raw HF ids
    for spec in REGISTRY.values():
        if spec.hf_id and spec.hf_id.lower() == key:
            return spec
    if "/" in name:
        return ModelSpec(name, name, 32768, notes="ad-hoc HF id (not in registry)")
    raise KeyError(f"unknown model {name!r}; known: {sorted(REGISTRY)}")
