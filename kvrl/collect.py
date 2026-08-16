"""CLI: collect full-cache traces for the simulator.

python -m kvrl.collect --config configs/collect.yaml [--split train] [--limit N] [key=value ...]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from kvrl.eval.corpus import Filler, load_corpus
from kvrl.eval.tasks import generate
from kvrl.models.hf_model import load_model
from kvrl.traces.collector import collect_trace
from kvrl.traces.storage import trace_index
from kvrl.utils.config import load_config
from kvrl.utils.tracker import start_run


def job_seed_offset(task: str, tokens: int) -> int:
    """Deterministic per-(task, length) seed offset (python's hash() is salted per process)."""
    import zlib

    return (zlib.crc32(f"{task}:{tokens}".encode()) % 1000) * 100


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/collect.yaml")
    ap.add_argument("--split", default=None, help="only this split (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="max traces per split (smoke tests)")
    ap.add_argument("overrides", nargs="*", help="dotted config overrides key=value")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    mcfg, ccfg = cfg["model"], cfg["collect"]
    model = load_model(mcfg["name"], device=mcfg.get("device"), dtype=mcfg.get("dtype"))
    print(
        f"[collect] model {model.info.name} on {model.info.device} {model.info.dtype} "
        f"({model.load_seconds:.1f}s)",
        flush=True,
    )
    corpus = load_corpus()
    out_root = Path(ccfg["out_dir"])
    with start_run("collect", cfg, device=str(model.device)) as run:
        totals = {}
        for split, scfg in ccfg["splits"].items():
            if args.split and split != args.split:
                continue
            out_dir = out_root / split
            out_dir.mkdir(parents=True, exist_ok=True)
            done = 0
            t_split = time.time()
            for job in scfg["jobs"]:
                task, tokens, n = job["task"], int(job["tokens"]), int(job["n"])
                for i in range(n):
                    if args.limit is not None and done >= args.limit:
                        break
                    seed = int(scfg["seed_base"]) + job_seed_offset(task, tokens) + i
                    trace_id = f"{task}_{tokens}_{seed}"
                    if (out_dir / f"{trace_id}.npz").exists():
                        done += 1
                        continue
                    filler = Filler(corpus, seed=seed)
                    inst = generate(task, 1, tokens, seed, filler, count_tokens=model.count_tokens)
                    if not inst:
                        print(f"[collect] {trace_id}: no instance generated", flush=True)
                        continue
                    t0 = time.time()
                    tr = collect_trace(
                        model,
                        inst[0],
                        trace_id,
                        max_new_tokens=int(ccfg["max_new_tokens"]),
                        chunk=int(ccfg["chunk"]),
                        out_dir=out_dir,
                        extra_meta={"split": split},
                    )
                    dt = time.time() - t0
                    done += 1
                    run.log(
                        split=split,
                        trace_id=trace_id,
                        T=tr.T,
                        K=tr.n_steps,
                        seconds=round(dt, 1),
                        correct=tr.meta["correct_full"],
                        bytes=(out_dir / f"{trace_id}.npz").stat().st_size,
                    )
                    print(
                        f"[collect] {split} {trace_id}: T={tr.T} K={tr.n_steps} {dt:.1f}s "
                        f"correct={tr.meta['correct_full']} "
                        f"text={tr.meta['generated_text'][:40]!r}",
                        flush=True,
                    )
            df = trace_index(out_dir, rebuild=True)
            totals[split] = {
                "n": len(df),
                "bytes": int(df["bytes"].sum()) if len(df) else 0,
                "seconds": round(time.time() - t_split, 1),
            }
            print(f"[collect] split {split}: {totals[split]}", flush=True)
        run.finish(totals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
