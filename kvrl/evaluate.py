"""CLI: evaluate controllers on the real model over the long-context task suite.

python -m kvrl.evaluate --config configs/evaluate.yaml [key=value ...]
"""

from __future__ import annotations

import argparse
import json
import sys

from kvrl.eval.runner import evaluate_real
from kvrl.models.hf_model import load_model
from kvrl.utils.config import load_config
from kvrl.utils.tracker import start_run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/evaluate.yaml")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    mcfg = cfg["model"]
    model = load_model(mcfg["name"], device=mcfg.get("device"), dtype=mcfg.get("dtype"))
    with start_run(
        "eval", cfg, device=str(model.device), run_id=args.run_id, seed=cfg["eval"].get("seed")
    ) as run:
        print(
            f"[eval] run {run.run_id} model {model.info.name} {model.info.dtype} {model.device}",
            flush=True,
        )
        rows_path = run.artifact_path("results.jsonl")
        f = open(rows_path, "a")

        def on_row(row):
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()

        df, summary = evaluate_real(model, cfg, log=lambda s: print(s, flush=True), on_row=on_row)
        f.close()
        df.to_parquet(run.artifact_path("results.parquet"), index=False)
        run.finish(summary)
        for rec in summary.get("table", []):
            acc = rec.get("accuracy", {}).get("mean")
            nll = rec["nll"].get("mean", float("nan"))
            print(
                f"[eval] {rec['controller']:10s} b={rec['budget_frac']:<5} n={rec['n']:3d} "
                f"acc={acc if acc is None else round(acc, 3)} nll={nll:.3f} "
                f"kv%={rec['kv_peak_frac']:.2f} model_s={rec['model_s']:.1f} "
                f"ctrl_s={rec['controller_s']:.2f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
