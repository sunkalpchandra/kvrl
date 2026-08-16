// Horizontal metric bar: value vs reference, both labelled. Neutral, monochrome, technical.
export function Bar({ label, value, max, text, tone = "zinc" }: { label: string; value: number; max: number; text: string; tone?: "zinc" | "emerald" | "sky" | "amber" }) {
  const w = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  const color = { zinc: "bg-zinc-400", emerald: "bg-emerald-400", sky: "bg-sky-400", amber: "bg-amber-400" }[tone];
  return (
    <div className="mb-2">
      <div className="flex items-baseline justify-between">
        <span className="label">{label}</span>
        <span className="mono text-xs text-zinc-300">{text}</span>
      </div>
      <div className="mt-1 h-1.5 w-full rounded bg-zinc-800">
        <div className={`h-1.5 rounded ${color}`} style={{ width: `${100 * w}%` }} />
      </div>
    </div>
  );
}

export function Kpi({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div>
      <div className="label">{label}</div>
      <div className="kpi">{value}</div>
      {sub && <div className="text-xs text-zinc-500">{sub}</div>}
    </div>
  );
}
