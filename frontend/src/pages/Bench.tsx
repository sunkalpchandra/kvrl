import { useEffect, useMemo, useState } from "react";
import { api, fmt } from "../api";
import { colorOf, Legend } from "../components/Scatter";

// Latency / memory vs context length per controller (medians over repeats) + the pure
// hardware curve: decode ms/token vs KV cache length with a full cache.
export function Bench() {
  const [data, setData] = useState<any>(null);
  const [active, setActive] = useState<Set<string>>(new Set());
  useEffect(() => { api.bench().then(d => { setData(d); setActive(new Set((d.rows ?? []).map((r: any) => r.controller))); }).catch(() => setData({ rows: [] })); }, []);
  const rows: any[] = useMemo(() => (data?.rows ?? []).filter((r: any) => r.ok && active.has(r.controller)), [data, active]);
  const series = useMemo(() => Array.from(new Set((data?.rows ?? []).map((r: any) => r.controller))) as string[], [data]);
  if (!data) return <div className="card text-sm text-zinc-500">loading…</div>;
  if (!data.rows?.length) return <div className="card text-sm text-zinc-500">No benchmark run yet — run <span className="mono">python -m kvrl.benchmark</span>.</div>;
  const contexts = Array.from(new Set(rows.map(r => r.context))).sort((a, b) => a - b);
  const Chart = ({ metric, label, fmtY }: { metric: string; label: string; fmtY: (v: number) => string }) => {
    const W = 380, H = 200, m = { l: 48, r: 10, t: 10, b: 30 };
    const ys = rows.map(r => r[metric]); const y1 = Math.max(...ys, 1e-9);
    const sx = (c: number) => m.l + (Math.log2(c) - Math.log2(contexts[0])) / Math.max(1e-9, Math.log2(contexts[contexts.length - 1]) - Math.log2(contexts[0])) * (W - m.l - m.r);
    const sy = (v: number) => H - m.b - (v / y1) * (H - m.t - m.b);
    return (<svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {contexts.map(c => <text key={c} x={sx(c)} y={H - m.b + 12} fontSize="9" fill="#71717a" textAnchor="middle">{c >= 1024 ? `${c / 1024}K` : c}</text>)}
      {[0, 0.5, 1].map(f => <g key={f}><line x1={m.l} x2={W - m.r} y1={sy(f * y1)} y2={sy(f * y1)} stroke="#27272a" /><text x={m.l - 4} y={sy(f * y1) + 3} fontSize="9" fill="#71717a" textAnchor="end">{fmtY(f * y1)}</text></g>)}
      <text x={12} y={(m.t + H - m.b) / 2} fontSize="10" fill="#a1a1aa" textAnchor="middle" transform={`rotate(-90 12 ${(m.t + H - m.b) / 2})`}>{label}</text>
      {series.filter(s => active.has(s)).map(s => {
        const pts = rows.filter(r => r.controller === s).sort((a, b) => a.context - b.context);
        return <g key={s}><path d={pts.map((p, i) => `${i ? "L" : "M"}${sx(p.context)},${sy(p[metric])}`).join(" ")} fill="none" stroke={colorOf(s)} strokeWidth="1.5" />{pts.map((p, i) => <circle key={i} cx={sx(p.context)} cy={sy(p[metric])} r="3" fill={colorOf(s)} />)}</g>;
      })}
    </svg>);
  };
  const curve: any[] = data.decode_curve ?? [];
  return (
    <div className="space-y-4">
      <div className="card">
        <div className="mb-2 flex items-baseline justify-between"><h3 className="text-sm font-medium">Latency and memory vs context length <span className="tag tag-real ml-2">real</span></h3><span className="text-xs text-zinc-500">run {data.run_id} · {data.meta?.device_info?.gpu ?? ""} · medians</span></div>
        <Legend series={series} active={active} onToggle={s => setActive(a => { const n = new Set(a); n.has(s) ? n.delete(s) : n.add(s); return n; })} />
        <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-3">
          <Chart metric="prefill_s_median" label="prefill (s)" fmtY={v => v.toFixed(1)} />
          <Chart metric="decode_ms_per_tok_median" label="decode ms / token" fmtY={v => v.toFixed(0)} />
          <Chart metric="kv_peak_frac" label="peak KV (fraction of full)" fmtY={v => v.toFixed(2)} />
        </div>
      </div>
      {curve.length > 0 && (<div className="card">
        <h3 className="mb-2 text-sm font-medium">Hardware curve: decode cost vs KV cache length (full cache)</h3>
        <table className="w-full text-xs mono"><thead className="text-zinc-500"><tr><th className="px-2 py-1 text-left font-normal">cache length</th><th className="px-2 py-1 text-left font-normal">decode ms/token (median)</th><th className="px-2 py-1 text-left font-normal">IQR</th><th className="px-2 py-1 text-left font-normal">KV bytes</th></tr></thead>
          <tbody>{curve.map((c: any) => <tr key={c.cache_len} className="border-t border-zinc-900"><td className="px-2 py-1">{c.cache_len}</td><td className="px-2 py-1">{c.decode_ms_per_tok_median.toFixed(1)}</td><td className="px-2 py-1">{c.p25.toFixed(1)}–{c.p75.toFixed(1)}</td><td className="px-2 py-1">{fmt.mb(c.kv_bytes)}</td></tr>)}</tbody></table>
      </div>)}
      <div className="card overflow-auto"><h3 className="mb-2 text-sm font-medium">All measurements</h3>
        <table className="w-full text-xs mono"><thead className="text-zinc-500"><tr>{["context", "controller", "budget", "prefill s", "decode ms/tok", "controller s", "compact s", "KV peak", "peak alloc MB"].map(h => <th key={h} className="px-2 py-1 text-left font-normal">{h}</th>)}</tr></thead>
          <tbody>{rows.sort((a, b) => a.context - b.context || a.controller.localeCompare(b.controller)).map((r, i) => <tr key={i} className="border-t border-zinc-900"><td className="px-2 py-1">{r.context}</td><td className="px-2 py-1">{r.controller}</td><td className="px-2 py-1">{fmt.pct(r.budget_frac)}</td><td className="px-2 py-1">{r.prefill_s_median?.toFixed(2)}</td><td className="px-2 py-1">{r.decode_ms_per_tok_median?.toFixed(1)}</td><td className="px-2 py-1">{r.controller_s_median?.toFixed(3)}</td><td className="px-2 py-1">{r.compact_s_median?.toFixed(3)}</td><td className="px-2 py-1">{fmt.pct(r.kv_peak_frac)}</td><td className="px-2 py-1">{r.peak_allocated_mb_median?.toFixed(0)}</td></tr>)}</tbody></table>
      </div>
    </div>
  );
}
