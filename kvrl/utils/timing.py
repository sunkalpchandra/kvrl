"""Timing helpers with device synchronisation baked in."""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from .device import synchronize


@contextmanager
def timed(device: torch.device | str, sink: dict | None = None, key: str = "elapsed_s"):
    """``with timed(dev, d, 'prefill_s'): ...`` — synchronised wall-clock seconds into ``d``."""
    synchronize(device)
    t0 = time.perf_counter()
    yield
    synchronize(device)
    dt = time.perf_counter() - t0
    if sink is not None:
        sink[key] = sink.get(key, 0.0) + dt


class Stopwatch:
    """Accumulate synchronised timings under named keys."""

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    @contextmanager
    def __call__(self, key: str):
        synchronize(self.device)
        t0 = time.perf_counter()
        yield
        synchronize(self.device)
        self.totals[key] = self.totals.get(key, 0.0) + time.perf_counter() - t0
        self.counts[key] = self.counts.get(key, 0) + 1

    def as_dict(self) -> dict[str, float]:
        return {f"{k}_s": v for k, v in self.totals.items()}

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()


@dataclass
class Samples:
    """Latency samples with robust summary statistics."""

    values: list[float] = field(default_factory=list)

    def add(self, v: float) -> None:
        self.values.append(v)

    def summary(self) -> dict[str, float]:
        v = sorted(self.values)
        if not v:
            return {}
        n = len(v)
        q = statistics.quantiles(v, n=4) if n >= 2 else [v[0], v[0], v[0]]
        return {
            "n": n,
            "median": statistics.median(v),
            "mean": statistics.fmean(v),
            "p25": q[0],
            "p75": q[2],
            "min": v[0],
            "max": v[-1],
            "std": statistics.pstdev(v) if n > 1 else 0.0,
        }
