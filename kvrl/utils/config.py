"""YAML config loading with dotted overrides and stable hashing.

A config is a plain nested dict. ``load_config`` supports::

    load_config("configs/train.yaml", overrides=["rl.lr=1e-4", "cache.budget=0.25"])

Values in overrides are parsed with YAML semantics (so ``0.25`` → float, ``true`` → bool).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def deep_update(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
        if not isinstance(cur, dict):
            raise ValueError(f"cannot descend into non-dict at {k!r} for {dotted!r}")
    cur[keys[-1]] = value


def get_dotted(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def parse_scalar(v: str) -> Any:
    """YAML scalar parsing, but tolerant of scientific notation like ``1e-4``
    (YAML 1.1 reads that as a string)."""
    out = yaml.safe_load(v)
    if isinstance(out, str) and _NUM_RE.match(out.strip()):
        return float(out)
    return out


def parse_override(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise ValueError(f"override must look like key.sub=value, got {s!r}")
    k, v = s.split("=", 1)
    return k.strip(), parse_scalar(v)


def load_config(path: str | Path | None, overrides: list[str] | None = None,
                defaults: dict | None = None) -> dict[str, Any]:
    """Load ``path`` (optional) on top of ``defaults`` (optional) then apply overrides.

    A config may declare ``_base: other.yaml`` (relative to itself) to inherit.
    """
    cfg: dict[str, Any] = copy.deepcopy(defaults or {})
    if path is not None:
        p = Path(path)
        loaded = load_yaml(p)
        base = loaded.pop("_base", None)
        if base:
            cfg = deep_update(cfg, load_config(p.parent / base))
        cfg = deep_update(cfg, loaded)
    for o in overrides or []:
        k, v = parse_override(o)
        set_dotted(cfg, k, v)
    return cfg


def config_hash(cfg: dict) -> str:
    """Stable 10-char hash of a config (used in run ids)."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def dump_yaml(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
