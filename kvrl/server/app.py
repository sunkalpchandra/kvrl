"""Dashboard API.

    uvicorn kvrl.server.app:app --reload --port 8000

Endpoints
---------
GET  /api/health
GET  /api/runs                  list tracked runs (meta + results summary)
GET  /api/runs/{run_id}         full run (config, meta, results, metrics)
GET  /api/runs/{run_id}/rows    per-prompt rows (eval results.parquet / bench.parquet)
GET  /api/pareto                latest eval run aggregated: quality vs KV% per controller/budget
GET  /api/checkpoints           trained policies with metadata
POST /api/demo                  run one prompt: full vs chosen controller (real model); returns
                                retention mask, decisions, timings, importance (takes seconds)
GET  /api/demo/stream?...       same as POST /api/demo, streamed as SSE events per decision

The server lazily loads the model on first demo request; the static frontend (frontend/dist)
is mounted at / when present.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from kvrl.utils.tracker import RUNS_DIR, list_runs, load_run

app = FastAPI(title="kvrl dashboard", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_MODEL_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"model": None, "engine": None, "controllers": {}}


def _runs_dir() -> Path:
    return Path(os.environ.get("KVRL_RUNS_DIR", RUNS_DIR))


@app.get("/api/health")
def health():
    return {"ok": True, "model_loaded": _STATE["model"] is not None}


@app.get("/api/runs")
def runs(kind: str | None = None):
    out = []
    for r in list_runs(_runs_dir(), kind=kind):
        meta = r.get("meta", {})
        res = r.get("results", {})
        out.append(
            {
                "run_id": meta.get("run_id"),
                "kind": meta.get("kind"),
                "created_at": meta.get("created_at"),
                "status": meta.get("status"),
                "commit": (meta.get("commit") or "")[:10],
                "device": (meta.get("device_info") or {}).get("gpu")
                or (meta.get("device_info") or {}).get("device"),
                "duration_s": meta.get("duration_s"),
                "summary_keys": list(res.keys())[:12] if isinstance(res, dict) else [],
            }
        )
    return out


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str):
    d = _runs_dir() / run_id
    if not d.exists():
        raise HTTPException(404, "run not found")
    r = load_run(d)
    return r


@app.get("/api/runs/{run_id}/rows")
def run_rows(run_id: str, limit: int = 5000):
    d = _runs_dir() / run_id
    if not d.exists():
        raise HTTPException(404, "run not found")
    import pandas as pd

    for name in ("results.parquet", "bench.parquet"):
        p = d / name
        if p.exists():
            df = pd.read_parquet(p)
            return json.loads(df.head(limit).to_json(orient="records"))
    p = d / "results.jsonl"
    if p.exists():
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        return rows[:limit]
    return []


@app.get("/api/pareto")
def pareto(run_id: str | None = None):
    """Aggregate an eval run into (controller, budget) → quality/memory/latency points."""
    import pandas as pd

    runs_ = list_runs(_runs_dir(), kind="eval")
    if run_id:
        runs_ = [r for r in runs_ if r["meta"]["run_id"] == run_id]
    runs_ = [r for r in runs_ if (Path(r["dir"]) / "results.parquet").exists()]
    if not runs_:
        return {"points": [], "run_id": None}
    # primary = the largest evaluation (most rows), consistent with scripts/make_report.py
    r = max(runs_, key=lambda x: (Path(x["dir"]) / "results.parquet").stat().st_size)
    df = pd.read_parquet(Path(r["dir"]) / "results.parquet")
    pts = []
    for (c, b), g in df.groupby(["controller", "budget_frac"]):
        acc = g["correct"].dropna()
        pts.append(
            {
                "controller": c,
                "budget_frac": float(b),
                "n": len(g),
                "accuracy": float(acc.astype(float).mean()) if len(acc) else None,
                "nll": float(g["nll"].mean()),
                "fidelity": float(g["fidelity"].dropna().mean())
                if g["fidelity"].notna().any()
                else None,
                "kv_peak_frac": float(g["kv_peak_frac"].mean()),
                "total_s": float(g["total_s"].median()),
                "model_s": float(g["model_s"].median()),
                "controller_s": float(g["controller_s"].median()),
                "decode_tok_per_s": float(g["decode_tok_per_s"].median()),
            }
        )
    by_task = []
    for (t, c, b), g in df.groupby(["task", "controller", "budget_frac"]):
        acc = g["correct"].dropna()
        by_task.append(
            {
                "task": t,
                "controller": c,
                "budget_frac": float(b),
                "n": len(g),
                "accuracy": float(acc.astype(float).mean()) if len(acc) else None,
                "nll": float(g["nll"].mean()),
                "kv_peak_frac": float(g["kv_peak_frac"].mean()),
            }
        )
    return {
        "run_id": r["meta"]["run_id"],
        "points": pts,
        "by_task": by_task,
        "meta": {k: r["meta"].get(k) for k in ("created_at", "commit", "device_info")},
    }


@app.get("/api/bench")
def bench_latest(run_id: str | None = None):
    """Latest benchmark run: per (context, controller, budget) medians + decode-vs-cache curve."""
    runs_ = [
        r
        for r in list_runs(_runs_dir(), kind="bench")
        if (Path(r["dir"]) / "bench.parquet").exists()
    ]
    if run_id:
        runs_ = [r for r in runs_ if r["meta"]["run_id"] == run_id]
    if not runs_:
        return {"run_id": None, "rows": [], "decode_curve": []}
    import pandas as pd

    r = runs_[-1]
    df = pd.read_parquet(Path(r["dir"]) / "bench.parquet")
    curve = (r.get("results") or {}).get("decode_curve", [])
    return {
        "run_id": r["meta"]["run_id"],
        "rows": json.loads(df.to_json(orient="records")),
        "decode_curve": curve,
        "meta": {k: r["meta"].get(k) for k in ("created_at", "commit", "device_info")},
    }


@app.get("/api/demo/snapshot")
def demo_snapshot():
    p = _runs_dir() / "demo_latest.json"
    if not p.exists():
        raise HTTPException(404, "no demo run yet")
    return json.loads(p.read_text())


@app.get("/api/checkpoints")
def checkpoints():
    import torch

    out = []
    for p in sorted(Path("checkpoints").glob("*.pt")):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            meta = ck.get("meta", {})
            out.append(
                {
                    "path": str(p),
                    "kind": ck.get("kind"),
                    "arch": ck.get("policy_kwargs", {}).get("arch"),
                    "meta": {k: v for k, v in meta.items() if k != "config"},
                }
            )
        except Exception as e:  # pragma: no cover
            out.append({"path": str(p), "error": repr(e)})
    return out


class DemoRequest(BaseModel):
    task: str = "needle"
    tokens: int = 2048
    budget_frac: float = 0.25
    controller: str = "rl"
    checkpoint: str = "checkpoints/ppo_mlp_v1_3.pt"
    max_new_tokens: int = 16
    seed: int = 7
    model: str = "qwen2.5-0.5b-instruct"
    text: str | None = None
    question: str | None = None


def _get_model(name: str):
    with _MODEL_LOCK:
        if _STATE["model"] is None or _STATE["model"].info.name != name:
            from kvrl.engine import InferenceEngine
            from kvrl.models.hf_model import load_model

            m = load_model(name)
            _STATE["model"] = m
            _STATE["engine"] = InferenceEngine(m, chunk_size=64, decide_every=64)
        return _STATE["model"], _STATE["engine"]


def _controller(name: str, checkpoint: str):
    from kvrl.controllers import make_controller

    if name == "rl" or name == "regressor":
        return make_controller(name, checkpoint=checkpoint)
    return make_controller(name)


def _run_demo(req: DemoRequest, emit=None) -> dict:
    from kvrl.engine import budget_from_fraction
    from kvrl.eval.corpus import Filler, load_corpus
    from kvrl.eval.tasks import generate, is_correct

    model, engine = _get_model(req.model)
    if req.text:
        prompt = req.text + ("\n\nQuestion: " + req.question if req.question else "")
        system = "You are a precise assistant. Answer using only the provided context."
        answers, crit = [], []
    else:
        filler = Filler(load_corpus(), seed=req.seed)
        inst = generate(req.task, 1, req.tokens, req.seed, filler, count_tokens=model.count_tokens)[
            0
        ]
        prompt, system, answers, crit = inst.prompt, inst.system, inst.answers, inst.critical_spans
    ids = model.encode_chat(prompt, system)
    n_prompt = int(ids.numel())
    budget = budget_from_fraction(req.budget_frac, n_prompt)
    # token strings for the UI (prompt only, capped)
    tok_strs = (
        [model.tokenizer.decode([int(t)]) for t in ids[: min(n_prompt, 16384)]]
        if model.tokenizer
        else []
    )
    crit_tokens: list[int] = []
    if crit:
        from kvrl.traces.collector import critical_token_mask

        cm = critical_token_mask(model, inst, ids)
        crit_tokens = [int(i) for i in cm.nonzero()[0]]
    out: dict[str, Any] = {
        "n_prompt": n_prompt,
        "budget": budget,
        "answers": answers,
        "critical_tokens": crit_tokens,
        "tokens": tok_strs,
        "model": model.info.name,
        "device": str(model.device),
        "kv_bytes_per_token": model.info.kv_bytes_per_token,
        "runs": {},
    }
    if emit:
        emit({"event": "start", "n_prompt": n_prompt, "budget": budget})
    for label, ctrl, b in [
        ("full", _controller("full", ""), 1 << 30),
        (req.controller, _controller(req.controller, req.checkpoint), budget),
    ]:

        def on_state(st, label=label):
            if emit and label != "full":
                emit(
                    {
                        "event": "state",
                        "controller": label,
                        "step": st.step,
                        "n": st.n,
                        "ctx_len": st.ctx_len,
                        "phase": st.phase,
                    }
                )

        res = engine.run(
            ids,
            ctrl,
            budget=b,
            max_new_tokens=req.max_new_tokens,
            record_importance=True,
            on_state=on_state,
        )
        rec = {
            "text": res.text,
            "generated_ids": res.generated_ids,
            "correct": is_correct(res.text, answers) if answers else None,
            "kv_bytes_peak": res.kv_bytes_peak,
            "kv_bytes_final": res.kv_bytes_final,
            "kv_bytes_full": res.kv_bytes_full,
            "peak_cache_len": res.peak_cache_len,
            "final_cache_len": res.final_cache_len,
            "n_evicted": res.n_evicted_total,
            "timings": res.timings,
            "memory": res.memory,
            "alive": res.alive,
            "stats_enabled": res.stats_enabled,
            "decisions": [
                {
                    "step": d.step,
                    "phase": d.phase,
                    "ctx_len": d.ctx_len,
                    "n_before": d.n_before,
                    "n_after": d.n_after,
                    "n_evicted": len(d.evicted_positions),
                    "evicted_positions": d.evicted_positions[:4096],
                    "controller_ms": 1000 * d.controller_s,
                    "compact_ms": 1000 * d.compact_s,
                    "importance": d.importance,
                }
                for d in res.decisions
            ],
        }
        if hasattr(ctrl, "describe"):
            rec["controller"] = ctrl.describe()
        out["runs"][label] = rec
        if emit:
            emit(
                {
                    "event": "done",
                    "controller": label,
                    "text": res.text,
                    "timings": res.timings,
                    "kv_bytes_peak": res.kv_bytes_peak,
                    "n_evicted": res.n_evicted_total,
                }
            )
    try:
        _runs_dir().mkdir(parents=True, exist_ok=True)
        (_runs_dir() / "demo_latest.json").write_text(json.dumps(out, default=str))
    except OSError as e:  # best effort: the snapshot is a convenience for the static dashboard
        out["snapshot_error"] = repr(e)
    return out


@app.post("/api/demo")
def demo(req: DemoRequest):
    try:
        return _run_demo(req)
    except FileNotFoundError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/demo/stream")
async def demo_stream(
    task: str = "needle",
    tokens: int = 2048,
    budget_frac: float = 0.25,
    controller: str = "rl",
    checkpoint: str = "checkpoints/ppo_mlp_v1_3.pt",
    max_new_tokens: int = 16,
    seed: int = 7,
    model: str = Query("qwen2.5-0.5b-instruct"),
):
    req = DemoRequest(
        task=task,
        tokens=tokens,
        budget_frac=budget_frac,
        controller=controller,
        checkpoint=checkpoint,
        max_new_tokens=max_new_tokens,
        seed=seed,
        model=model,
    )
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def emit(ev):
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    def worker():
        try:
            result = _run_demo(req, emit=emit)
            emit({"event": "result", "result": result})
        except Exception as e:  # pragma: no cover
            emit({"event": "error", "error": repr(e)})
        emit(None)

    threading.Thread(target=worker, daemon=True).start()

    async def gen():
        while True:
            ev = await queue.get()
            if ev is None:
                break
            yield f"data: {json.dumps(ev, default=str)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.exists():  # pragma: no cover
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="static")
