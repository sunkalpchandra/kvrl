"""Lightweight local experiment tracker (no external service).

Every run gets ``runs/<run_id>/`` containing:

- ``config.yaml``   the fully-resolved config
- ``meta.json``     commit, dirty flag, device info, seed, timestamps, python/torch versions
- ``metrics.jsonl`` one JSON object per logged step
- ``results.json``  final summary written by :meth:`Run.finish`
- arbitrary artifacts saved via :meth:`Run.artifact_path`

The dashboard and README tables read from these files; nothing else is a source of
truth for numbers.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .config import config_hash, dump_yaml
from .device import device_info

RUNS_DIR = Path(os.environ.get("KVRL_RUNS_DIR", "runs"))


def git_info(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _json_default(o: Any):
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


class Run:
    """A single tracked run. Use :func:`start_run`."""

    def __init__(self, run_id: str, run_dir: Path, config: dict, meta: dict):
        self.run_id = run_id
        self.dir = run_dir
        self.config = config
        self.meta = meta
        self._metrics_f = open(self.dir / "metrics.jsonl", "a")
        self._t0 = time.time()

    def log(self, step: int | None = None, **metrics: Any) -> None:
        rec = {"t": round(time.time() - self._t0, 3)}
        if step is not None:
            rec["step"] = step
        rec.update(metrics)
        self._metrics_f.write(json.dumps(rec, default=_json_default) + "\n")
        self._metrics_f.flush()

    def artifact_path(self, name: str) -> Path:
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_json(self, name: str, obj: Any) -> Path:
        p = self.artifact_path(name)
        with open(p, "w") as f:
            json.dump(obj, f, indent=2, default=_json_default)
        return p

    def finish(self, results: dict | None = None, status: str = "finished") -> Path:
        self.meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.meta["duration_s"] = round(time.time() - self._t0, 3)
        self.meta["status"] = status
        with open(self.dir / "meta.json", "w") as f:
            json.dump(self.meta, f, indent=2, default=_json_default)
        p = self.dir / "results.json"
        with open(p, "w") as f:
            json.dump(results or {}, f, indent=2, default=_json_default)
        self._metrics_f.close()
        return p

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.finish({"error": repr(exc)}, status="failed")
            return False
        if not (self.dir / "results.json").exists():
            self.finish({})
        return False


def start_run(
    kind: str,
    config: dict,
    *,
    seed: int | None = None,
    device: str | None = None,
    run_id: str | None = None,
    runs_dir: Path | None = None,
    tags: dict | None = None,
) -> Run:
    """Create a run directory and record provenance. ``kind`` e.g. 'train', 'eval', 'bench'."""
    root = Path(runs_dir or RUNS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rid = run_id or f"{stamp}-{kind}-{config_hash(config)}"
    run_dir = root / rid
    run_dir.mkdir(parents=True, exist_ok=False)
    dump_yaml(config, run_dir / "config.yaml")
    meta: dict[str, Any] = {
        "run_id": rid,
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "python": platform.python_version(),
        "argv": sys.argv,
        "config_hash": config_hash(config),
        "tags": tags or {},
        **git_info(),
    }
    if device is not None:
        try:
            meta["device_info"] = device_info(device)
        except Exception as e:  # pragma: no cover
            meta["device_info"] = {"error": repr(e)}
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)
    return Run(rid, run_dir, config, meta)


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Read a run back (meta, config, results, metrics list)."""
    d = Path(run_dir)
    out: dict[str, Any] = {"dir": str(d)}
    for name in ("meta.json", "results.json"):
        p = d / name
        if p.exists():
            with open(p) as f:
                out[name.split(".")[0]] = json.load(f)
    p = d / "metrics.jsonl"
    if p.exists():
        with open(p) as f:
            out["metrics"] = [json.loads(line) for line in f if line.strip()]
    p = d / "config.yaml"
    if p.exists():
        import yaml

        with open(p) as f:
            out["config"] = yaml.safe_load(f)
    return out


def list_runs(runs_dir: Path | None = None, kind: str | None = None) -> list[dict[str, Any]]:
    root = Path(runs_dir or RUNS_DIR)
    if not root.exists():
        return []
    runs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d / "meta.json").exists():
            continue
        r = load_run(d)
        if kind is None or r.get("meta", {}).get("kind") == kind:
            runs.append(r)
    return runs
