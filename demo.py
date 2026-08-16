#!/usr/bin/env python
"""One-click demo: full-cache vs adaptive-cache inference on a real long-context prompt.

    python demo.py                       # needle-in-a-haystack at ~4K tokens, RL policy if trained
    python demo.py --tokens 8192 --budget 0.25 --controller h2o
    python demo.py --file mydoc.txt --question "What is ...?"

Everything printed is measured in this process: no cached or hard-coded numbers.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from kvrl.controllers import make_controller
from kvrl.engine import InferenceEngine, budget_from_fraction
from kvrl.eval.corpus import Filler, load_corpus
from kvrl.eval.tasks import generate, is_correct
from kvrl.models.hf_model import load_model
from kvrl.utils.device import device_info

console = Console()


def retention_strip(alive: list[bool], width: int = 96) -> Text:
    """Downsample the per-token alive mask into a coloured strip (green kept, grey evicted)."""
    n = len(alive)
    t = Text()
    if n == 0:
        return t
    per = max(1, n // width)
    for i in range(0, n, per):
        block = alive[i : i + per]
        frac = sum(block) / len(block)
        ch = "█" if frac > 0.66 else ("▓" if frac > 0.33 else "░")
        style = "green" if frac > 0.66 else ("yellow" if frac > 0.33 else "grey42")
        t.append(ch, style=style)
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
    ap.add_argument(
        "--tokens", type=int, default=4096, help="context length for the synthetic task"
    )
    ap.add_argument(
        "--task", default="needle", choices=["needle", "kv", "multihop", "dependency", "code"]
    )
    ap.add_argument(
        "--budget", type=float, default=0.25, help="cache budget as a fraction of the prompt"
    )
    ap.add_argument(
        "--controller",
        default="auto",
        help="rl|h2o|window|snapkv|random|keynorm|auto (rl if a checkpoint exists)",
    )
    ap.add_argument("--checkpoint", default="checkpoints/ppo_mlp_v1_3.pt")
    ap.add_argument("--file", default=None, help="use this text file as the context")
    ap.add_argument("--question", default=None, help="question to ask about --file")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    console.rule("[bold]ADAPTIVE KV CACHE — live demo")
    with console.status("loading model..."):
        model = load_model(args.model, device=args.device)
    info = model.info
    di = device_info(model.device)
    console.print(
        f"Model [bold]{info.name}[/] · {info.n_layers} layers · {info.n_heads}/{info.n_kv_heads} heads · "
        f"{info.dtype} · {di.get('gpu', di.get('cpu', model.device))} · "
        f"KV = {info.kv_bytes_per_token:,} B/token"
    )

    # ------------------------------------------------------------------ prompt
    if args.file:
        text = Path(args.file).read_text(errors="ignore")
        question = args.question or "Summarise the text above in one sentence."
        prompt = text + "\n\nQuestion: " + question
        answers: list[str] = []
        system = "You are a precise assistant. Answer using only the provided context."
        crit = []
    else:
        filler = Filler(load_corpus(), seed=args.seed)
        inst = generate(
            args.task, 1, args.tokens, args.seed, filler, count_tokens=model.count_tokens
        )[0]
        prompt, answers, system, crit = inst.prompt, inst.answers, inst.system, inst.critical_spans
    ids = model.encode_chat(prompt, system)
    n_prompt = int(ids.numel())
    budget = budget_from_fraction(args.budget, n_prompt)
    console.print(
        f"Context: [bold]{n_prompt:,}[/] tokens · budget {args.budget:.0%} → {budget:,} slots"
        + (f" · task [bold]{args.task}[/] (answer: {answers[0]})" if answers else "")
    )

    ctrl_name = args.controller
    if ctrl_name == "auto":
        ctrl_name = "rl" if Path(args.checkpoint).exists() else "h2o"
        if ctrl_name == "h2o":
            console.print("[yellow]no trained policy checkpoint found → using H2O heuristic[/]")
    ctrl = (
        make_controller(ctrl_name, checkpoint=args.checkpoint)
        if ctrl_name == "rl"
        else make_controller(ctrl_name)
    )
    engine = InferenceEngine(model, chunk_size=64, decide_every=64)

    # ------------------------------------------------------------------ runs
    results = {}
    for label, c, b in [
        ("Full cache", make_controller("full"), 1 << 30),
        (f"{ctrl_name} cache", ctrl, budget),
    ]:
        with console.status(f"running {label}..."):
            t0 = time.time()
            res = engine.run(
                ids, c, budget=b, max_new_tokens=args.max_new_tokens, record_importance=True
            )
            res.wall = time.time() - t0  # type: ignore[attr-defined]
        results[label] = res

    # ------------------------------------------------------------------ report
    full, adapt = list(results.values())
    table = Table(title="Full cache vs adaptive cache (measured now)", show_lines=False)
    table.add_column("")
    table.add_column("Full cache", justify="right")
    table.add_column(f"{ctrl_name}", justify="right")
    table.add_column("Δ", justify="right")

    def row(name, a, b, fmt="{:.2f}", better="lower"):
        d = b - a
        sign = "[green]" if (d < 0) == (better == "lower") else "[red]"
        table.add_row(name, fmt.format(a), fmt.format(b), f"{sign}{fmt.format(d)}[/]")

    row("KV memory peak (MB)", full.kv_bytes_peak / 2**20, adapt.kv_bytes_peak / 2**20)
    row("KV memory final (MB)", full.kv_bytes_final / 2**20, adapt.kv_bytes_final / 2**20)
    row("Peak cache length", full.peak_cache_len, adapt.peak_cache_len, "{:.0f}")
    row("Prefill (s)", full.timings["prefill_s"], adapt.timings["prefill_s"])
    row(
        "Decode (ms/token)",
        1000 * full.timings["decode_s"] / max(1, len(full.generated_ids)),
        1000 * adapt.timings["decode_s"] / max(1, len(adapt.generated_ids)),
    )
    row("Controller (s)", full.timings["controller_s"], adapt.timings["controller_s"], "{:.3f}")
    row("Cache ops (s)", full.timings["compact_s"], adapt.timings["compact_s"], "{:.3f}")
    row("Total (s)", full.timings["total_s"], adapt.timings["total_s"])
    row("Tokens evicted", full.n_evicted_total, adapt.n_evicted_total, "{:.0f}", better="higher")
    console.print(table)
    console.print(
        Panel(Text(full.text.strip()[:300]), title="Full-cache answer", border_style="grey50")
    )
    console.print(
        Panel(Text(adapt.text.strip()[:300]), title=f"{ctrl_name} answer", border_style="cyan")
    )
    if answers:
        console.print(
            f"Correct: full={is_correct(full.text, answers)}  {ctrl_name}={is_correct(adapt.text, answers)}"
            f"  (expected {answers[0]!r})"
        )
    console.print(
        f"Output fidelity vs full cache: {sum(a == b for a, b in zip(full.generated_ids, adapt.generated_ids)) / max(1, len(full.generated_ids)):.0%} tokens agree"
    )

    console.print()
    console.print("[bold]TOKEN RETENTION[/] (prompt + generation; green kept, grey evicted)")
    console.print(retention_strip(adapt.alive))
    if crit:
        # mark critical tokens
        console.print(
            f"critical spans in prompt: {len(crit)}  ·  attention stats path: {'on' if adapt.stats_enabled else 'off'}"
        )
    kept = sum(adapt.alive)
    console.print(
        f"kept {kept:,} / {len(adapt.alive):,} tokens ({kept / max(1, len(adapt.alive)):.1%}) · "
        f"{adapt.n_decisions} decisions"
    )
    if adapt.decisions:
        last = adapt.decisions[-1]
        console.print(
            f"last decision: step {last.step} evicted {len(last.evicted_positions)} "
            f"(controller {1000 * last.controller_s:.1f} ms, compaction {1000 * last.compact_s:.1f} ms)"
        )
    if hasattr(ctrl, "describe"):
        console.print(f"controller: {ctrl.describe()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
