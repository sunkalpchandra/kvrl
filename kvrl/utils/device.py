"""Device abstraction so the same code measures correctly on CUDA, Apple MPS and CPU.

Every latency measurement in kvrl goes through :func:`synchronize` and every memory
measurement through :func:`memory_stats`; nothing else in the codebase touches
``torch.cuda`` / ``torch.mps`` directly.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

import torch


def resolve_device(preference: str | None = None) -> torch.device:
    """Pick a device. ``preference`` may be 'auto', 'cpu', 'mps', 'cuda' or 'cuda:N'."""
    pref = (preference or os.environ.get("KVRL_DEVICE") or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(pref)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    return dev


def synchronize(device: torch.device | str) -> None:
    """Block until all queued kernels on ``device`` finished (no-op on CPU)."""
    t = torch.device(device).type
    if t == "cuda":
        torch.cuda.synchronize()
    elif t == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device | str) -> None:
    t = torch.device(device).type
    if t == "cuda":
        torch.cuda.empty_cache()
    elif t == "mps":
        torch.mps.empty_cache()


def reset_peak_memory(device: torch.device | str) -> None:
    t = torch.device(device).type
    if t == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # MPS has no peak-reset API; PeakTracker below handles it by sampling.


@dataclass
class MemoryStats:
    allocated_bytes: int
    reserved_bytes: int
    peak_bytes: int | None  # None where the backend cannot report a peak

    def as_dict(self) -> dict:
        return {
            "allocated_mb": self.allocated_bytes / 2**20,
            "reserved_mb": self.reserved_bytes / 2**20,
            "peak_mb": None if self.peak_bytes is None else self.peak_bytes / 2**20,
        }


def memory_stats(device: torch.device | str) -> MemoryStats:
    """Current allocator statistics for ``device``."""
    t = torch.device(device).type
    if t == "cuda":
        return MemoryStats(
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
            torch.cuda.max_memory_allocated(),
        )
    if t == "mps":
        return MemoryStats(
            torch.mps.current_allocated_memory(),
            torch.mps.driver_allocated_memory(),
            None,
        )
    return MemoryStats(0, 0, None)


class PeakTracker:
    """Track peak allocated bytes across a region on backends without native peak stats.

    Usage::

        with PeakTracker(device) as pk:
            ...; pk.sample()   # call at interesting points (after forward, after eviction)
        pk.peak_bytes
    """

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        self.peak_bytes = 0
        self.start_bytes = 0

    def sample(self) -> int:
        cur = memory_stats(self.device).allocated_bytes
        if self.device.type == "cuda":
            cur = max(cur, torch.cuda.max_memory_allocated())
        self.peak_bytes = max(self.peak_bytes, cur)
        return cur

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.start_bytes = memory_stats(self.device).allocated_bytes
        self.peak_bytes = self.start_bytes
        return self

    def __exit__(self, *exc):
        self.sample()
        return False


def device_info(device: torch.device | str) -> dict:
    """Human-readable hardware description recorded with every run."""
    dev = torch.device(device)
    info: dict = {
        "device": str(dev),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        info.update(
            gpu=p.name,
            total_memory_gb=round(p.total_memory / 2**30, 2),
            cuda=torch.version.cuda,
        )
    elif dev.type == "mps":
        info.update(
            gpu="Apple MPS",
            chip=_apple_chip_name(),
            recommended_max_memory_gb=round(torch.mps.recommended_max_memory() / 2**30, 2),
        )
    else:
        info.update(cpu=platform.processor() or _apple_chip_name(), threads=torch.get_num_threads())
    return info


def _apple_chip_name() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:  # pragma: no cover - non-mac
        return platform.processor()


def supports_bf16(device: torch.device | str) -> bool:
    """Whether bfloat16 matmul actually runs on this device (MPS support varies by OS)."""
    dev = torch.device(device)
    if dev.type == "cpu":
        return True
    try:
        a = torch.ones(4, 4, device=dev, dtype=torch.bfloat16)
        b = (a @ a).float().sum().item()
        return abs(b - 64.0) < 1e-3
    except Exception:
        return False


def pick_dtype(device: torch.device | str, requested: str | None = None) -> torch.dtype:
    """Map a config dtype string to a torch dtype that works on ``device``."""
    dev = torch.device(device)
    name = (requested or "auto").lower()
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name in table:
        dt = table[name]
        if dt is torch.bfloat16 and not supports_bf16(dev):
            raise RuntimeError(f"bfloat16 requested but not supported on {dev}")
        return dt
    if name != "auto":
        raise ValueError(f"unknown dtype {requested!r}")
    if dev.type == "cpu":
        return torch.float32
    if dev.type == "mps":
        # Measured 2026-08-16 on Apple M2 / torch 2.13: fp16 is ~2.6x faster than bf16
        # (prefill 1751 vs 668 tok/s, decode 23 vs 50 ms/tok on Qwen2.5-0.5B). See D-004.
        return torch.float16
    return torch.bfloat16 if supports_bf16(dev) else torch.float16
