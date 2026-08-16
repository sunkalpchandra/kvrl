import type { Decision } from "../api";

// Per-decision trace: step, phase, cache before → after, evicted count, controller/compaction ms.
export function DecisionTrace({ decisions, onPick, picked }: { decisions: Decision[]; onPick?: (d: Decision) => void; picked?: number }) {
  return (
    <div className="max-h-72 overflow-auto rounded border border-zinc-800/80">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-[#101318] text-zinc-500">
          <tr>{["step", "phase", "ctx", "cache", "evicted", "controller", "compact"].map(h => <th key={h} className="px-2 py-1 text-left font-normal">{h}</th>)}</tr>
        </thead>
        <tbody className="mono">
          {decisions.map(d => (
            <tr key={d.step} onClick={() => onPick?.(d)} className={`cursor-pointer border-t border-zinc-900 hover:bg-zinc-900/60 ${picked === d.step ? "bg-zinc-900" : ""}`}>
              <td className="px-2 py-1">{d.step}</td>
              <td className="px-2 py-1">{d.phase === 0 ? "prefill" : "decode"}</td>
              <td className="px-2 py-1">{d.ctx_len}</td>
              <td className="px-2 py-1">{d.n_before} → {d.n_after}</td>
              <td className="px-2 py-1 text-zinc-300">{d.n_evicted}</td>
              <td className="px-2 py-1">{d.controller_ms.toFixed(1)} ms</td>
              <td className="px-2 py-1">{d.compact_ms.toFixed(1)} ms</td>
            </tr>))}
        </tbody>
      </table>
    </div>
  );
}
