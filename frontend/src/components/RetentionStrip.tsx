import { useEffect, useMemo, useRef, useState } from "react";

// Every token of the context as one cell: green kept, grey evicted, amber = task-critical
// (outlined). Click a cell for the token's detail (position, text, status, importance).
export type TokenDetail = { index: number; text: string; alive: boolean; critical: boolean; importance: number | null; evictedAt: number | null };

export function RetentionStrip({ alive, tokens, critical, decisions, importance, height = 64, onSelect }:
  { alive: boolean[]; tokens: string[]; critical: number[]; decisions: { step: number; evicted_positions: number[] }[];
    importance?: Map<number, number>; height?: number; onSelect?: (d: TokenDetail) => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<TokenDetail | null>(null);
  const critSet = useMemo(() => new Set(critical), [critical]);
  const evictedAt = useMemo(() => {
    const m = new Map<number, number>();
    for (const d of decisions) for (const p of d.evicted_positions) m.set(p, d.step);
    return m;
  }, [decisions]);
  const n = alive.length;

  useEffect(() => {
    const cv = ref.current; if (!cv || n === 0) return;
    const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = height;
    cv.width = W * dpr; cv.height = H * dpr;
    const ctx = cv.getContext("2d")!; ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);
    const rows = n > 4096 ? 4 : n > 1024 ? 2 : 1;
    const perRow = Math.ceil(n / rows);
    const cw = W / perRow, rh = H / rows;
    for (let i = 0; i < n; i++) {
      const r = Math.floor(i / perRow), c = i % perRow;
      const x = c * cw, y = r * rh;
      const imp = importance?.get(i);
      if (alive[i]) {
        const a = imp == null ? 0.85 : 0.35 + 0.65 * imp;
        ctx.fillStyle = `rgba(52, 211, 153, ${a})`;
      } else ctx.fillStyle = "rgba(63, 63, 70, 0.55)";
      ctx.fillRect(x, y + 1, Math.max(cw, 0.5), rh - 2);
      if (critSet.has(i)) { ctx.fillStyle = "rgba(251, 191, 36, 0.95)"; ctx.fillRect(x, y, Math.max(cw, 1), 2); ctx.fillRect(x, y + rh - 2, Math.max(cw, 1), 2); }
    }
  }, [alive, critSet, importance, height, n]);

  const detailAt = (e: React.MouseEvent<HTMLCanvasElement>): TokenDetail | null => {
    const cv = ref.current; if (!cv || n === 0) return null;
    const rect = cv.getBoundingClientRect();
    const rows = n > 4096 ? 4 : n > 1024 ? 2 : 1; const perRow = Math.ceil(n / rows);
    const c = Math.floor(((e.clientX - rect.left) / rect.width) * perRow);
    const r = Math.floor(((e.clientY - rect.top) / rect.height) * rows);
    const i = r * perRow + c; if (i < 0 || i >= n) return null;
    return { index: i, text: tokens[i] ?? "", alive: alive[i], critical: critSet.has(i), importance: importance?.get(i) ?? null, evictedAt: evictedAt.get(i) ?? null };
  };
  return (
    <div>
      <canvas ref={ref} className="w-full cursor-crosshair rounded" style={{ height }}
        onMouseMove={(e) => setHover(detailAt(e))} onMouseLeave={() => setHover(null)}
        onClick={(e) => { const d = detailAt(e); if (d && onSelect) onSelect(d); }} />
      <div className="mt-1 flex justify-between text-[11px] text-zinc-500">
        <span>token 0</span>
        <span className="mono">{hover ? `#${hover.index} ${JSON.stringify(hover.text)} · ${hover.alive ? "kept" : `evicted @ step ${hover.evictedAt}`}${hover.critical ? " · critical" : ""}` : "hover for token detail · click to pin"}</span>
        <span>token {n - 1}</span>
      </div>
    </div>
  );
}
