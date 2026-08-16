import { useEffect, useMemo, useState } from "react";
import { api, fmt, type DemoResult, type Decision } from "../api";
import { Bar, Kpi } from "../components/Bars";
import { RetentionStrip, type TokenDetail } from "../components/RetentionStrip";
import { DecisionTrace } from "../components/DecisionTrace";

// Live demo: run one prompt through the real model with a full cache and with the chosen
// controller; every number on this page comes from that run (or a labelled snapshot).
export function Live({ isStatic }: { isStatic: boolean }) {
  const [params, setParams] = useState({ task: "needle", tokens: 2048, budget_frac: 0.25, controller: "rl", max_new_tokens: 16, seed: 7 });
  const [result, setResult] = useState<DemoResult | null>(null);
  const [status, setStatus] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [picked, setPicked] = useState<TokenDetail | null>(null);
  const [pickedStep, setPickedStep] = useState<number | undefined>();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { if (isStatic) api.demoSnapshot().then(setResult).catch(() => setResult(null)); }, [isStatic]);

  const run = () => {
    setRunning(true); setError(null); setStatus("starting…"); setResult(null);
    const stop = api.demoStream(params, (ev) => {
      if (ev.event === "start") setStatus(`prompt ${ev.n_prompt} tokens · budget ${ev.budget}`);
      else if (ev.event === "state") setStatus(`${ev.controller}: step ${ev.step} · cache ${ev.n} · ctx ${ev.ctx_len} (${ev.phase ? "decode" : "prefill"})`);
      else if (ev.event === "done") setStatus(`${ev.controller} done in ${fmt.s(ev.timings.total_s)}`);
      else if (ev.event === "result") { setResult(ev.result); setRunning(false); stop(); }
      else if (ev.event === "error") { setError(ev.error); setRunning(false); stop(); }
    });
  };

  const adaptKey = result ? Object.keys(result.runs).find(k => k !== "full") : undefined;
  const full = result?.runs.full, adapt = adaptKey ? result?.runs[adaptKey] : undefined;
  const importance = useMemo(() => {
    const m = new Map<number, number>();
    if (!adapt) return m;
    // importance recorded per kept slot at the last decision; map slots → positions via alive order
    const last = adapt.decisions[adapt.decisions.length - 1];
    if (!last?.importance) return m;
    const kept: number[] = []; adapt.alive.forEach((a, i) => { if (a) kept.push(i); });
    const vals = last.importance; const lo = Math.min(...vals), hi = Math.max(...vals);
    kept.slice(0, vals.length).forEach((pos, i) => m.set(pos, hi > lo ? (vals[i] - lo) / (hi - lo) : 0.5));
    return m;
  }, [adapt]);

  return (
    <div className="space-y-4">
      <div className="card flex flex-wrap items-end gap-3">
        <label className="text-xs text-zinc-400">task<br /><select className="input" value={params.task} onChange={e => setParams({ ...params, task: e.target.value })}>{["needle", "kv", "multihop", "dependency", "code"].map(t => <option key={t}>{t}</option>)}</select></label>
        <label className="text-xs text-zinc-400">context tokens<br /><select className="input" value={params.tokens} onChange={e => setParams({ ...params, tokens: +e.target.value })}>{[1024, 2048, 4096, 8192, 16384].map(t => <option key={t} value={t}>{t}</option>)}</select></label>
        <label className="text-xs text-zinc-400">budget<br /><select className="input" value={params.budget_frac} onChange={e => setParams({ ...params, budget_frac: +e.target.value })}>{[0.5, 0.25, 0.125].map(b => <option key={b} value={b}>{Math.round(100 * b)}%</option>)}</select></label>
        <label className="text-xs text-zinc-400">controller<br /><select className="input" value={params.controller} onChange={e => setParams({ ...params, controller: e.target.value })}>{["rl", "h2o", "snapkv", "window", "random", "keynorm", "regressor"].map(c => <option key={c}>{c}</option>)}</select></label>
        <label className="text-xs text-zinc-400">seed<br /><input className="input w-20" type="number" value={params.seed} onChange={e => setParams({ ...params, seed: +e.target.value })} /></label>
        <button className="btn" disabled={running || isStatic} onClick={run}>{running ? "running…" : "run on real model"}</button>
        <span className="mono text-xs text-zinc-500">{isStatic ? "static snapshot mode (backend not running)" : status}</span>
        {error && <span className="text-xs text-red-400">{error}</span>}
      </div>

      {result && full && adapt && (<>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {[["Full cache", full, "zinc"], [`${adaptKey} cache`, adapt, "emerald"]].map(([title, r, tone]) => {
            const rr = r as typeof full;
            const dec = rr.timings.decode_s / Math.max(1, rr.generated_ids?.length ?? rr.timings.decode_s / 0.05);
            return (
              <div className="card" key={title as string}>
                <div className="mb-3 flex items-baseline justify-between"><h3 className="text-sm font-medium text-zinc-100">{title as string}</h3><span className="tag tag-real">real</span></div>
                <Bar label="KV memory (peak)" value={rr.kv_bytes_peak} max={full.kv_bytes_full} text={`${fmt.mb(rr.kv_bytes_peak)} · ${fmt.pct(rr.kv_bytes_peak / full.kv_bytes_full)}`} tone={tone as any} />
                <Bar label="Prefill" value={rr.timings.prefill_s} max={Math.max(full.timings.prefill_s, adapt.timings.prefill_s)} text={fmt.s(rr.timings.prefill_s)} tone={tone as any} />
                <Bar label="Decode / token" value={dec} max={Math.max(full.timings.decode_s, adapt.timings.decode_s) / Math.max(1, full.generated_ids?.length ?? 1)} text={fmt.s(dec)} tone={tone as any} />
                <Bar label="Controller + cache ops" value={rr.timings.controller_s + rr.timings.compact_s} max={Math.max(0.001, adapt.timings.controller_s + adapt.timings.compact_s)} text={fmt.s(rr.timings.controller_s + rr.timings.compact_s)} tone={tone as any} />
                <div className="mt-3 grid grid-cols-3 gap-2">
                  <Kpi label="tokens kept" value={fmt.pct(rr.alive.filter(Boolean).length / rr.alive.length)} sub={`${rr.alive.filter(Boolean).length} / ${rr.alive.length}`} />
                  <Kpi label="evicted" value={String(rr.n_evicted)} sub={`${rr.decisions.length} decisions`} />
                  <Kpi label="answer" value={rr.correct == null ? "–" : rr.correct ? "correct" : "wrong"} sub={result.answers[0] ? `expected ${result.answers[0]}` : ""} />
                </div>
                <div className="mt-3 rounded bg-black/30 p-2 mono text-xs text-zinc-300">{rr.text}</div>
              </div>);
          })}
        </div>

        <div className="card">
          <div className="mb-2 flex items-baseline justify-between"><h3 className="text-sm font-medium">Token retention · {adaptKey}</h3><span className="text-xs text-zinc-500">green kept (brighter = higher predicted importance) · grey evicted · amber = task-critical</span></div>
          <RetentionStrip alive={adapt.alive} tokens={result.tokens} critical={result.critical_tokens} decisions={adapt.decisions} importance={importance} onSelect={setPicked} />
          {picked && (
            <div className="mt-3 grid grid-cols-2 gap-3 rounded border border-zinc-800 p-3 text-xs md:grid-cols-5">
              <div><div className="label">token</div><div className="mono">#{picked.index}</div></div>
              <div><div className="label">text</div><div className="mono">{JSON.stringify(picked.text)}</div></div>
              <div><div className="label">status</div><div>{picked.alive ? "KEEP" : `EVICT @ step ${picked.evictedAt}`}</div></div>
              <div><div className="label">predicted importance</div><div className="mono">{picked.importance == null ? "–" : picked.importance.toFixed(2)}</div></div>
              <div><div className="label">critical</div><div>{picked.critical ? "yes" : "no"}</div></div>
            </div>)}
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="card"><h3 className="mb-2 text-sm font-medium">Decision trace · {adaptKey}</h3><DecisionTrace decisions={adapt.decisions} picked={pickedStep} onPick={(d: Decision) => setPickedStep(d.step)} /></div>
          <div className="card">
            <h3 className="mb-2 text-sm font-medium">Run facts</h3>
            <table className="w-full text-xs mono"><tbody>
              {[["model", result.model], ["device", result.device], ["KV bytes / token", result.kv_bytes_per_token.toLocaleString()], ["prompt tokens", result.n_prompt], ["budget (slots)", result.budget], ["stats path", adapt.stats_enabled ? "on (attention statistics captured)" : "off"], ["controller", JSON.stringify(adapt.controller ?? {})]].map(([k, v]) => (<tr key={k as string} className="border-t border-zinc-900"><td className="py-1 text-zinc-500">{k as string}</td><td className="py-1 text-right">{String(v)}</td></tr>))}
            </tbody></table>
          </div>
        </div>
      </>)}
    </div>
  );
}
