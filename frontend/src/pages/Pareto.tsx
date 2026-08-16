import { useEffect, useMemo, useState } from "react";
import { api, fmt, type Pareto as ParetoT } from "../api";
import { Legend, Scatter, type Pt } from "../components/Scatter";

// Pareto frontier: KV memory (peak, fraction of full) vs quality, per controller; toggle series.
export function ParetoPage() {
  const [data, setData] = useState<ParetoT | null>(null);
  const [metric, setMetric] = useState<"accuracy" | "fidelity" | "nll">("accuracy");
  const [active, setActive] = useState<Set<string>>(new Set());
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.pareto().then(d => { setData(d); setActive(new Set(d.points.map(p => p.controller))); }).catch(e => setErr(String(e))); }, []);
  const series = useMemo(() => Array.from(new Set(data?.points.map(p => p.controller) ?? [])), [data]);
  const pts: Pt[] = useMemo(() => (data?.points ?? []).filter(p => active.has(p.controller)).map(p => {
    const y = metric === "nll" ? p.nll : (p[metric] ?? NaN);
    return { x: p.kv_peak_frac, y, label: p.controller === "full" ? "100%" : `${Math.round(100 * p.budget_frac)}%`, series: p.controller };
  }).filter(p => !Number.isNaN(p.y)), [data, active, metric]);
  if (err) return <div className="card text-sm text-red-400">{err}</div>;
  if (!data) return <div className="card text-sm text-zinc-500">loading…</div>;
  if (!data.points.length) return <div className="card text-sm text-zinc-500">No evaluation run yet — run <span className="mono">python -m kvrl.evaluate</span>.</div>;
  const yLabel = metric === "accuracy" ? "task accuracy" : metric === "fidelity" ? "output fidelity vs full cache (ROUGE-L)" : "NLL (lower is better)";
  return (
    <div className="space-y-4">
      <div className="card">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div><h3 className="text-sm font-medium">Quality vs KV memory <span className="tag tag-real ml-2">real</span></h3><div className="text-xs text-zinc-500">eval run {data.run_id} · {data.meta?.device_info?.gpu ?? ""} · commit {(data.meta?.commit ?? "").slice(0, 8)}</div></div>
          <div className="flex gap-2">{(["accuracy", "fidelity", "nll"] as const).map(m => <button key={m} className={`btn ${metric === m ? "border-zinc-400" : ""}`} onClick={() => setMetric(m)}>{m}</button>)}</div>
        </div>
        <Legend series={series} active={active} onToggle={s => setActive(a => { const n = new Set(a); n.has(s) ? n.delete(s) : n.add(s); return n; })} />
        <div className="mt-3"><Scatter pts={pts} xLabel="peak KV memory (fraction of full cache)" yLabel={yLabel} yDomain={metric === "nll" ? undefined : [0, 1]} /></div>
      </div>
      <div className="card overflow-auto">
        <h3 className="mb-2 text-sm font-medium">Per controller × budget</h3>
        <table className="w-full text-xs mono">
          <thead className="text-zinc-500"><tr>{["controller", "budget", "n", "accuracy", "fidelity", "nll", "KV peak", "model s", "controller s", "decode tok/s"].map(h => <th key={h} className="px-2 py-1 text-left font-normal">{h}</th>)}</tr></thead>
          <tbody>{data.points.sort((a, b) => a.controller.localeCompare(b.controller) || b.budget_frac - a.budget_frac).map((p, i) => (
            <tr key={i} className="border-t border-zinc-900"><td className="px-2 py-1">{p.controller}</td><td className="px-2 py-1">{p.controller === "full" ? "100%" : fmt.pct(p.budget_frac)}</td><td className="px-2 py-1">{p.n}</td><td className="px-2 py-1">{fmt.pct(p.accuracy, 1)}</td><td className="px-2 py-1">{fmt.num(p.fidelity)}</td><td className="px-2 py-1">{fmt.num(p.nll, 3)}</td><td className="px-2 py-1">{fmt.pct(p.kv_peak_frac)}</td><td className="px-2 py-1">{fmt.num(p.model_s)}</td><td className="px-2 py-1">{fmt.num(p.controller_s, 3)}</td><td className="px-2 py-1">{fmt.num(p.decode_tok_per_s, 1)}</td></tr>))}
          </tbody></table>
      </div>
    </div>
  );
}
