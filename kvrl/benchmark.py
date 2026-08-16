"""CLI: latency/memory benchmark on the real model.

python -m kvrl.benchmark --config configs/benchmark.yaml [key=value ...]
"""

from __future__ import annotations

import argparse
import json
import sys

from kvrl.bench.harness import bench, decode_vs_cache_length
from kvrl.models.hf_model import load_model
from kvrl.utils.config import load_config
from kvrl.utils.tracker import start_run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    mcfg = cfg["model"]
    model = load_model(mcfg["name"], device=mcfg.get("device"), dtype=mcfg.get("dtype"))
    with start_run("bench", cfg, device=str(model.device), run_id=args.run_id) as run:
        print(
            f"[bench] run {run.run_id} model {model.info.name} {model.info.dtype} {model.device}",
            flush=True,
        )
        f = open(run.artifact_path("bench.jsonl"), "a")
        df, summary = bench(
            model,
            cfg,
            log=lambda s: print(s, flush=True),
            on_row=lambda r: (f.write(json.dumps(r, default=str) + "\n"), f.flush()),
        )
        f.close()
        df.to_parquet(run.artifact_path("bench.parquet"), index=False)
        hw = cfg["benchmark"].get("decode_curve_lengths")
        if hw:
            curve = decode_vs_cache_length(
                model,
                [int(x) for x in hw],
                repeats=int(cfg["benchmark"].get("repeats", 3)),
                log=lambda s: print(s, flush=True),
            )
            curve.to_parquet(run.artifact_path("decode_curve.parquet"), index=False)
            summary["decode_curve"] = curve.to_dict(orient="records")
        run.finish(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
