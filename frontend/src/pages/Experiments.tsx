import { useEffect, useState } from "react";
import { api, fmt, type RunSummary } from "../api";

// Experiment explorer: every tracked run (train / eval / bench / collect) with its provenance,
// and per-prompt rows for eval runs filterable by task / controller / budget.
export function Experiments() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [detail, setDetail] = useState<any>(null);
  const [rows, setRows] = useState<any[]>([]);
  const [filter, setFilter] = useState({ task: "", controller: "", budget: "" });
  useEffect(() => { api.runs().then(setRuns).catch(() => setRuns([])); }, []);
  useEffect(() => { if (!sel) return; api.run(sel).then(setDetail).catch(() => setDetail(null)); api.rows(sel).then(setRows).catch(() => setRows([])); }, [sel]);
  const tasks = Array.from(new Set(rows.map(r => r.task).filter(Boolean)));
  const ctrls = Array.from(new Set(rows.map(r => r.controller).filter(Boolean)));
  const budgets = Array.from(new Set(rows.map(r => r.budget_frac).filter(x => x != null))).sort((a, b) => b - a);
  const shown = rows.filter(r => (!filter.task || r.task === filter.task) && (!filter.controller || r.controller === filter.controller) && (!filter.budget || String(r.budget_frac) === filter.budget));
  const cols = rows.length && "context" in rows[0] ? ["context", "controller", "budget_frac", "prefill_s_median", "decode_ms_per_tok_median", "controller_s_median", "compact_s_median", "kv_peak_frac", "peak_allocated_mb_median"] : ["task", "n_prompt", "controller", "budget_frac", "correct", "nll", "fidelity", "kv_peak_frac", "model_s", "controller_s", "text"];
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[320px_1fr]">
      <div className="card max-h-[75vh] overflow-auto">
        <h3 className="mb-2 text-sm font-medium">Runs</h3>
        {runs.length === 0 && <div className="text-xs text-zinc-500">no runs found</div>}
        {runs.slice().reverse().map(r => (
          <button key={r.run_id} onClick={() => setSel(r.run_id)} className={`mb-1 block w-full rounded border px-2 py-1.5 text-left text-xs ${sel === r.run_id ? "border-zinc-500 bg-zinc-900" : "border-zinc-800 hover:border-zinc-600"}`}>
            <div className="flex justify-between"><span className="text-zinc-200">{r.kind}</span><span className="text-zinc-500">{r.status}</span></div>
            <div className="mono text-[10px] text-zinc-500">{r.run_id}</div>
            <div className="text-[10px] text-zinc-500">{r.device ?? ""} · {r.commit} · {r.duration_s ? fmt.s(r.duration_s) : ""}</div>
          </button>))}
      </div>
      <div className="space-y-4">
        {detail && (<div className="card">
          <h3 className="text-sm font-medium">{detail.meta?.kind} · <span className="mono text-xs text-zinc-400">{detail.meta?.run_id}</span></h3>
          <div className="mt-1 text-xs text-zinc-500">commit {detail.meta?.commit} {detail.meta?.dirty ? "(dirty)" : ""} · seed {String(detail.meta?.seed)} · {detail.meta?.device_info?.gpu ?? detail.meta?.device_info?.device} · torch {detail.meta?.device_info?.torch}</div>
          <details className="mt-2 text-xs"><summary className="cursor-pointer text-zinc-400">config + results (json)</summary><pre className="mono mt-2 max-h-64 overflow-auto rounded bg-black/30 p-2 text-[10px]">{JSON.stringify({ config: detail.config, results: detail.results }, null, 1)}</pre></details>
        </div>)}
        {rows.length > 0 && (<div className="card">
          <div className="mb-2 flex flex-wrap gap-2 text-xs">
            <select className="input" value={filter.task} onChange={e => setFilter({ ...filter, task: e.target.value })}><option value="">all tasks</option>{tasks.map(t => <option key={t}>{t}</option>)}</select>
            <select className="input" value={filter.controller} onChange={e => setFilter({ ...filter, controller: e.target.value })}><option value="">all controllers</option>{ctrls.map(t => <option key={t}>{t}</option>)}</select>
            <select className="input" value={filter.budget} onChange={e => setFilter({ ...filter, budget: e.target.value })}><option value="">all budgets</option>{budgets.map(b => <option key={b} value={String(b)}>{fmt.pct(b)}</option>)}</select>
            <span className="self-center text-zinc-500">{shown.length} rows</span>
          </div>
          <div className="max-h-[55vh] overflow-auto"><table className="w-full text-[11px] mono"><thead className="sticky top-0 bg-[#101318] text-zinc-500"><tr>{cols.map(c => <th key={c} className="px-2 py-1 text-left font-normal">{c}</th>)}</tr></thead>
            <tbody>{shown.slice(0, 500).map((r, i) => (<tr key={i} className="border-t border-zinc-900">{cols.map(c => <td key={c} className="max-w-[240px] truncate px-2 py-0.5">{typeof r[c] === "number" ? (Number.isInteger(r[c]) ? r[c] : r[c].toFixed(3)) : String(r[c] ?? "")}</td>)}</tr>))}</tbody></table></div>
        </div>)}
      </div>
    </div>
  );
}
