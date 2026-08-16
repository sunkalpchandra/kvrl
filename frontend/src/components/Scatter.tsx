// Minimal SVG scatter/line chart for the Pareto view: x = KV memory (fraction of full),
// y = quality (accuracy or fidelity or -NLL). One colour per controller, budget as label.
const PALETTE: Record<string, string> = { full: "#a1a1aa", rl: "#34d399", h2o: "#60a5fa", snapkv: "#f472b6", window: "#fbbf24", random: "#71717a", keynorm: "#c084fc", regressor: "#22d3ee", tova: "#fb923c", hybrid: "#a3e635" };
export const colorOf = (c: string) => PALETTE[c] || "#e4e4e7";

export type Pt = { x: number; y: number; label: string; series: string; size?: number };
export function Scatter({ pts, xLabel, yLabel, width = 620, height = 320, xDomain, yDomain, connect = true }:
  { pts: Pt[]; xLabel: string; yLabel: string; width?: number; height?: number; xDomain?: [number, number]; yDomain?: [number, number]; connect?: boolean }) {
  const m = { l: 48, r: 16, t: 12, b: 36 };
  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  const [x0, x1] = xDomain ?? [0, Math.max(1, ...xs)];
  const [y0, y1] = yDomain ?? [Math.min(...ys, 0), Math.max(...ys, 1)];
  const sx = (x: number) => m.l + ((x - x0) / (x1 - x0 || 1)) * (width - m.l - m.r);
  const sy = (y: number) => height - m.b - ((y - y0) / (y1 - y0 || 1)) * (height - m.t - m.b);
  const series = Array.from(new Set(pts.map(p => p.series)));
  const ticks = (a: number, b: number, n = 5) => Array.from({ length: n + 1 }, (_, i) => a + (i * (b - a)) / n);
  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} className="overflow-visible">
      {ticks(x0, x1).map((t, i) => (<g key={`x${i}`}><line x1={sx(t)} x2={sx(t)} y1={m.t} y2={height - m.b} stroke="#27272a" /><text x={sx(t)} y={height - m.b + 14} fontSize="10" fill="#71717a" textAnchor="middle">{(100 * t).toFixed(0)}%</text></g>))}
      {ticks(y0, y1).map((t, i) => (<g key={`y${i}`}><line x1={m.l} x2={width - m.r} y1={sy(t)} y2={sy(t)} stroke="#27272a" /><text x={m.l - 6} y={sy(t) + 3} fontSize="10" fill="#71717a" textAnchor="end">{t.toFixed(2)}</text></g>))}
      <text x={(m.l + width - m.r) / 2} y={height - 4} fontSize="11" fill="#a1a1aa" textAnchor="middle">{xLabel}</text>
      <text x={12} y={(m.t + height - m.b) / 2} fontSize="11" fill="#a1a1aa" textAnchor="middle" transform={`rotate(-90 12 ${(m.t + height - m.b) / 2})`}>{yLabel}</text>
      {series.map(s => {
        const sp = pts.filter(p => p.series === s).sort((a, b) => a.x - b.x);
        const path = sp.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.y)}`).join(" ");
        return (<g key={s}>
          {connect && sp.length > 1 && <path d={path} fill="none" stroke={colorOf(s)} strokeWidth="1.2" strokeOpacity="0.6" />}
          {sp.map((p, i) => (<g key={i}><circle cx={sx(p.x)} cy={sy(p.y)} r={p.size ?? 4} fill={colorOf(s)} /><text x={sx(p.x) + 6} y={sy(p.y) - 6} fontSize="9" fill="#a1a1aa">{p.label}</text></g>))}
        </g>);
      })}
    </svg>
  );
}

export function Legend({ series, active, onToggle }: { series: string[]; active: Set<string>; onToggle: (s: string) => void }) {
  return (<div className="flex flex-wrap gap-2">{series.map(s => (
    <button key={s} onClick={() => onToggle(s)} className={`flex items-center gap-1.5 rounded border px-2 py-0.5 text-xs ${active.has(s) ? "border-zinc-600 text-zinc-200" : "border-zinc-800 text-zinc-600"}`}>
      <span className="inline-block h-2 w-2 rounded-full" style={{ background: colorOf(s), opacity: active.has(s) ? 1 : 0.3 }} />{s}
    </button>))}</div>);
}
