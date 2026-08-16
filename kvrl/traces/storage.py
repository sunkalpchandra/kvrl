"""Trace file format: one compressed .npz per prompt + a parquet index per split."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Trace:
    trace_id: str
    token_ids: np.ndarray  # int32 [T]  (prompt + generated)
    n_prompt: int
    n_gen: int
    chunk: int
    attn_mean: np.ndarray  # float16 [K, T]  row k: A_k(j) layer-mean fraction (Σ_j = 1)
    attn_lmax: np.ndarray  # float16 [K, T]  layer-max of head-mean fraction
    step_end: np.ndarray  # int32 [K]  number of tokens in cache after step k's chunk
    step_phase: np.ndarray  # int8 [K]  0 prefill / 1 decode
    key_norm: np.ndarray  # float16 [T]
    value_norm: np.ndarray  # float16 [T]
    adj_key_cos: np.ndarray  # float16 [T]
    gen_logprob: np.ndarray  # float16 [G]  full-cache log-prob of generated tokens
    critical_mask: np.ndarray  # bool [T]
    meta: dict = field(default_factory=dict)
    answer_mask: np.ndarray | None = None  # bool [T]: answer tokens inside the critical span

    @property
    def n_steps(self) -> int:
        return int(self.attn_mean.shape[0])

    @property
    def T(self) -> int:
        return int(self.token_ids.shape[0])

    def nbytes(self) -> int:
        return sum(
            getattr(self, k).nbytes
            for k in (
                "token_ids",
                "attn_mean",
                "attn_lmax",
                "step_end",
                "step_phase",
                "key_norm",
                "value_norm",
                "adj_key_cos",
                "gen_logprob",
                "critical_mask",
            )
        )


def save_trace(tr: Trace, directory: str | Path) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{tr.trace_id}.npz"
    np.savez_compressed(
        p,
        token_ids=tr.token_ids.astype(np.int32),
        n_prompt=np.int32(tr.n_prompt),
        n_gen=np.int32(tr.n_gen),
        chunk=np.int32(tr.chunk),
        attn_mean=tr.attn_mean.astype(np.float16),
        attn_lmax=tr.attn_lmax.astype(np.float16),
        step_end=tr.step_end.astype(np.int32),
        step_phase=tr.step_phase.astype(np.int8),
        key_norm=tr.key_norm.astype(np.float16),
        value_norm=tr.value_norm.astype(np.float16),
        adj_key_cos=tr.adj_key_cos.astype(np.float16),
        gen_logprob=tr.gen_logprob.astype(np.float16),
        critical_mask=tr.critical_mask.astype(bool),
        meta=np.array(json.dumps(tr.meta, default=str)),
        **({"answer_mask": tr.answer_mask.astype(bool)} if tr.answer_mask is not None else {}),
    )
    return p


def load_trace(path: str | Path) -> Trace:
    z = np.load(path, allow_pickle=False)
    return Trace(
        trace_id=Path(path).stem,
        token_ids=z["token_ids"],
        n_prompt=int(z["n_prompt"]),
        n_gen=int(z["n_gen"]),
        chunk=int(z["chunk"]),
        attn_mean=z["attn_mean"],
        attn_lmax=z["attn_lmax"],
        step_end=z["step_end"],
        step_phase=z["step_phase"],
        key_norm=z["key_norm"],
        value_norm=z["value_norm"],
        adj_key_cos=z["adj_key_cos"],
        gen_logprob=z["gen_logprob"],
        critical_mask=z["critical_mask"],
        meta=json.loads(str(z["meta"])),
        answer_mask=z["answer_mask"] if "answer_mask" in z.files else None,
    )


def trace_index(directory: str | Path, rebuild: bool = False):
    """Return (and cache as parquet) a DataFrame of trace metadata in ``directory``."""
    import pandas as pd

    d = Path(directory)
    idx = d / "index.parquet"
    if idx.exists() and not rebuild:
        return pd.read_parquet(idx)
    rows = []
    for p in sorted(d.glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        rows.append(
            {
                "trace_id": p.stem,
                "path": str(p),
                "n_prompt": int(z["n_prompt"]),
                "n_gen": int(z["n_gen"]),
                "n_steps": int(z["attn_mean"].shape[0]),
                "task": meta.get("task"),
                "seed": meta.get("seed"),
                "target_tokens": meta.get("target_tokens"),
                "correct_full": meta.get("correct_full"),
                "n_critical": int(z["critical_mask"].sum()),
                "bytes": p.stat().st_size,
            }
        )
    df = pd.DataFrame(rows)
    if rows:
        df.to_parquet(idx, index=False)
    return df
