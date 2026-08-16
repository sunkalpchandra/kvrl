"""CLI: train the PPO cache-eviction policy in the simulator.

python -m kvrl.train --config configs/train.yaml [key=value ...]
"""

from __future__ import annotations

import argparse
import sys

from kvrl.rl.train import train
from kvrl.utils.config import load_config
from kvrl.utils.tracker import start_run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    with start_run(
        "train", cfg, seed=cfg.get("seed", 0), device=cfg.get("device", "cpu"), run_id=args.run_id
    ) as run:
        print(f"[train] run {run.run_id}", flush=True)
        results = train(cfg, run, log=lambda *a: print(*a, flush=True))
        run.finish(results)
        print(f"[train] done: {results}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
