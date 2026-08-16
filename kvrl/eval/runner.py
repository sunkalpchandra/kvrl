"""Real-model evaluation: controllers × budgets × tasks × context lengths.

For every (task instance, budget, controller) the engine runs the real model; we record
task accuracy, NLL of the natural continuation (lm), output fidelity vs the full-cache
generation of the same prompt, KV bytes (peak / final), latency breakdown and controller
overhead. Rows go to ``runs/<id>/results.parquet``; the summary has paired comparisons.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pandas as pd

from kvrl.controllers import make_controller
from kvrl.engine import InferenceEngine, budget_from_fraction
from kvrl.eval.corpus import Filler, load_corpus
from kvrl.eval.metrics import paired_difference, prefix_agreement, rouge_l, summarize
from kvrl.eval.tasks import TaskInstance, generate, is_correct
from kvrl.models.hf_model import HFCausalLM


def build_controllers(specs: list[dict]) -> dict[str, object]:
    out = {}
    for spec in specs:
        spec = dict(spec)
        name = spec.pop("name")
        label = spec.pop("label", name)
        out[label] = make_controller(name, **spec)
    return out


def evaluate_real(
    model: HFCausalLM,
    cfg: dict,
    log: Callable[[str], None] = print,
    on_row: Callable[[dict], None] | None = None,
) -> tuple[pd.DataFrame, dict]:
    ecfg = cfg["eval"]
    corpus = load_corpus()
    controllers = build_controllers(ecfg["controllers"])
    engine = InferenceEngine(
        model,
        chunk_size=int(ecfg.get("chunk", 64)),
        decide_every=int(ecfg.get("decide_every", 64)),
        n_sink=int(ecfg.get("n_sink", 4)),
    )
    budgets = [float(b) for b in ecfg["budget_fracs"]]
    rows: list[dict] = []
    t_all = time.time()
    for job in ecfg["jobs"]:
        task, tokens, n = job["task"], int(job["tokens"]), int(job["n"])
        seed = int(job.get("seed", ecfg.get("seed", 2000)))
        filler = Filler(corpus, seed=seed)
        instances = generate(task, n, tokens, seed, filler, count_tokens=model.count_tokens)
        for inst in instances:
            rows.extend(
                _eval_instance(model, engine, inst, controllers, budgets, ecfg, log, on_row)
            )
    df = pd.DataFrame(rows)
    summary = summarize_results(df)
    summary["seconds"] = round(time.time() - t_all, 1)
    return df, summary


def _eval_instance(model, engine, inst: TaskInstance, controllers, budgets, ecfg, log, on_row):
    ids = model.encode_chat(inst.prompt, inst.system) if inst.system else model.encode(inst.prompt)
    n_prompt = int(ids.numel())
    max_new = int(ecfg.get("max_new_tokens", inst.max_new_tokens or 16))
    forced = None
    if inst.task == "lm" and inst.continuation:
        forced = model.encode(inst.continuation)[: int(ecfg.get("lm_continuation_tokens", 64))]
    rows = []
    # full-cache reference first (fidelity target)
    full = controllers.get("full") or make_controller("full")
    ref = engine.run(ids, full, budget=1 << 30, max_new_tokens=max_new, forced_ids=forced)
    ref_row = _row(inst, "full", 1.0, 1 << 30, n_prompt, ref, ref, forced is not None)
    rows.append(ref_row)
    _emit(log, on_row, ref_row)
    for bf in budgets:
        budget = budget_from_fraction(
            bf,
            n_prompt,
            chunk=engine.chunk_size,
            min_tokens=int(ecfg.get("min_budget_tokens", 128)),
        )
        for label, ctrl in controllers.items():
            if label == "full":
                continue
            res = engine.run(
                ids,
                ctrl,
                budget=budget,
                max_new_tokens=max_new,
                forced_ids=forced,
                episode=int(inst.meta.get("seed", 0)),
            )
            row = _row(inst, label, bf, budget, n_prompt, res, ref, forced is not None)
            rows.append(row)
            _emit(log, on_row, row)
    return rows


def _emit(log, on_row, row):
    if on_row is not None:
        on_row(row)
    log(
        f"[eval] {row['task']:10s} n={row['n_prompt']:5d} b={row['budget_frac']:<5} "
        f"{row['controller']:10s} acc={row['correct']} nll={row['nll']:.3f} "
        f"fid={row['fidelity']:.2f} kv%={row['kv_peak_frac']:.2f} "
        f"model={row['model_s']:.1f}s ctrl={row['controller_s']:.2f}s"
    )


def _row(inst, label, bf, budget, n_prompt, res, ref, forced: bool) -> dict:
    gen = res.generated_ids
    correct = is_correct(res.text, inst.answers) if inst.answers else None
    fid = rouge_l(gen, ref.generated_ids) if not forced else float("nan")
    pre = prefix_agreement(gen, ref.generated_ids) if not forced else float("nan")
    return {
        "task": inst.task,
        "seed": inst.meta.get("seed"),
        "target_tokens": inst.meta.get("target_tokens"),
        "n_prompt": n_prompt,
        "controller": label,
        "budget_frac": bf,
        "budget": budget,
        "correct": None if correct is None else bool(correct),
        "answer": (inst.answers or [None])[0],
        "text": res.text[:200],
        "nll": res.nll,
        "n_generated": len(gen),
        "fidelity": fid,
        "prefix_agreement": pre,
        "kv_peak_frac": res.kv_bytes_peak / max(1, res.kv_bytes_full),
        "kv_final_frac": res.kv_bytes_final / max(1, res.kv_bytes_full),
        "kv_bytes_peak": res.kv_bytes_peak,
        "kv_bytes_full": res.kv_bytes_full,
        "peak_cache_len": res.peak_cache_len,
        "n_evicted": res.n_evicted_total,
        "n_decisions": res.n_decisions,
        "prefill_s": res.timings["prefill_s"],
        "decode_s": res.timings["decode_s"],
        "model_s": res.timings["model_s"],
        "controller_s": res.timings["controller_s"],
        "compact_s": res.timings["compact_s"],
        "total_s": res.timings["total_s"],
        "decode_tok_per_s": res.timings["decode_tok_per_s"],
        "peak_allocated_mb": res.memory.get("peak_allocated_mb"),
        "stats_enabled": res.stats_enabled,
        "critical_spans": len(inst.critical_spans),
        "depth": inst.meta.get("depth"),
    }


def summarize_results(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {}
    out: dict = {"n_rows": len(df)}
    grp = df.groupby(["controller", "budget_frac"])
    table = []
    for (c, b), g in grp:
        rec = {"controller": c, "budget_frac": b, "n": len(g)}
        acc = g["correct"].dropna()
        if len(acc):
            rec["accuracy"] = summarize(acc.astype(float).tolist())
        rec["nll"] = summarize(g["nll"].dropna().tolist())
        fid = g["fidelity"].dropna()
        if len(fid):
            rec["fidelity"] = summarize(fid.tolist())
        rec["kv_peak_frac"] = float(g["kv_peak_frac"].mean())
        rec["model_s"] = float(g["model_s"].median())
        rec["controller_s"] = float(g["controller_s"].median())
        rec["compact_s"] = float(g["compact_s"].median())
        rec["total_s"] = float(g["total_s"].median())
        rec["decode_tok_per_s"] = float(g["decode_tok_per_s"].median())
        table.append(rec)
    out["table"] = table
    # paired comparisons per prompt: accuracy and fidelity on graded tasks, NLL on lm only
    pairs = []
    key = ["task", "seed", "target_tokens"]
    full = df[df.controller == "full"].set_index(key)

    def _pair(label, b, vs, a_df, b_df):
        common = a_df.index.intersection(b_df.index)
        lm = [k for k in common if k[0] == "lm"]
        graded = [k for k in common if k[0] != "lm"]
        if len(lm) >= 2:
            pairs.append(
                {
                    "controller": label,
                    "budget_frac": b,
                    "vs": vs,
                    "metric": "nll_lm",
                    **paired_difference(
                        a_df.loc[lm, "nll"].tolist(),
                        b_df.loc[lm, "nll"].tolist(),
                        lower_is_better=True,
                    ),
                }
            )
        if len(graded) >= 2:
            pairs.append(
                {
                    "controller": label,
                    "budget_frac": b,
                    "vs": vs,
                    "metric": "fidelity",
                    **paired_difference(
                        a_df.loc[graded, "fidelity"].tolist(), b_df.loc[graded, "fidelity"].tolist()
                    ),
                }
            )
            acc_a, acc_b = a_df.loc[graded, "correct"], b_df.loc[graded, "correct"]
            ok = acc_a.notna() & acc_b.notna()
            if int(ok.sum()) >= 2:
                pairs.append(
                    {
                        "controller": label,
                        "budget_frac": b,
                        "vs": vs,
                        "metric": "accuracy",
                        **paired_difference(
                            acc_a[ok].astype(float).tolist(), acc_b[ok].astype(float).tolist()
                        ),
                    }
                )

    for (c, b), g in grp:
        if c == "full":
            continue
        gi = g.set_index(key)
        _pair(c, b, "full", gi, full)
        if "rl" in df.controller.values and c != "rl":
            rl = df[(df.controller == "rl") & (df.budget_frac == b)].set_index(key)
            _pair("rl", b, c, rl, gi)
    out["paired"] = pairs
    return out


__all__ = ["build_controllers", "evaluate_real", "summarize_results"]
