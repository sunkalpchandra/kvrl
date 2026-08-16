#!/usr/bin/env python
"""Re-measure the decode-vs-cache-length hardware curve and store it in a bench run.

    python scripts/measure_decode_curve.py --run <bench_run_id> [--lengths 512,1024,...]

Writes results.json['decode_curve'] (+ a provenance note) so the report picks it up.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvrl.bench.harness import decode_vs_cache_length
from kvrl.models.hf_model import load_model
from kvrl.utils.tracker import git_info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192,16384")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    d = Path("runs") / args.run
    res_p = d / "results.json"
    res = json.loads(res_p.read_text())
    model = load_model("qwen2.5-0.5b-instruct")
    curve = decode_vs_cache_length(model, [int(x) for x in args.lengths.split(",")], repeats=args.repeats,
                                   log=lambda s: print(s, flush=True))
    res["decode_curve"] = curve.to_dict(orient="records")
    res["decode_curve_note"] = {"measured_at": datetime.now(timezone.utc).isoformat(),
                                "method": "chunked prefill (256) + MPS pool release; scripts/measure_decode_curve.py",
                                **git_info()}
    curve.to_parquet(d / "decode_curve.parquet", index=False)
    res_p.write_text(json.dumps(res, indent=2, default=str))
    print(f"updated {res_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
