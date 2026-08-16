// API client with a static "demo mode" fallback (GitHub Pages): when the backend is not
// reachable, JSON snapshots under /demo/*.json (exported by scripts/export_demo_snapshot.py)
// are used and every view is labelled as a snapshot.

export type ParetoPoint = {
  controller: string; budget_frac: number; n: number; accuracy: number | null; nll: number;
  fidelity: number | null; kv_peak_frac: number; total_s: number; model_s: number;
  controller_s: number; decode_tok_per_s: number;
};
export type Pareto = { run_id: string | null; points: ParetoPoint[]; by_task?: any[]; meta?: any };
export type RunSummary = { run_id: string; kind: string; created_at: string; status: string; commit: string; device?: string; duration_s?: number };
export type Decision = { step: number; phase: number; ctx_len: number; n_before: number; n_after: number; n_evicted: number; evicted_positions: number[]; controller_ms: number; compact_ms: number; importance: number[] | null };
export type DemoRun = { text: string; correct: boolean | null; kv_bytes_peak: number; kv_bytes_final: number; kv_bytes_full: number; peak_cache_len: number; final_cache_len: number; n_evicted: number; timings: Record<string, number>; memory: Record<string, number>; alive: boolean[]; decisions: Decision[]; stats_enabled: boolean; controller?: any };
export type DemoResult = { n_prompt: number; budget: number; answers: string[]; critical_tokens: number[]; tokens: string[]; model: string; device: string; kv_bytes_per_token: number; runs: Record<string, DemoRun> };

let staticMode: boolean | null = null;
const base = (import.meta.env.BASE_URL || "/").replace(/\/$/, "");

async function detect(): Promise<boolean> {
  if (staticMode !== null) return staticMode;
  try {
    const r = await fetch("/api/health", { signal: AbortSignal.timeout(1500) });
    staticMode = !r.ok;
  } catch { staticMode = true; }
  return staticMode;
}
export async function isStatic() { return detect(); }

async function getJson<T>(apiPath: string, snapshot: string): Promise<T> {
  if (await detect()) {
    const r = await fetch(`${base}/demo/${snapshot}`);
    if (!r.ok) throw new Error(`snapshot ${snapshot} missing`);
    return r.json();
  }
  const r = await fetch(apiPath);
  if (!r.ok) throw new Error(`${apiPath}: ${r.status}`);
  return r.json();
}

export const api = {
  runs: () => getJson<RunSummary[]>("/api/runs", "runs.json"),
  run: (id: string) => getJson<any>(`/api/runs/${id}`, `run_${id}.json`),
  rows: (id: string) => getJson<any[]>(`/api/runs/${id}/rows`, `rows_${id}.json`),
  pareto: () => getJson<Pareto>("/api/pareto", "pareto.json"),
  checkpoints: () => getJson<any[]>("/api/checkpoints", "checkpoints.json"),
  bench: () => getJson<any>("/api/bench", "bench.json"),
  demoSnapshot: () => getJson<DemoResult>("/api/demo/snapshot", "demo.json"),
  demoStream: (params: Record<string, string | number>, onEvent: (ev: any) => void): (() => void) => {
    const qs = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString();
    const es = new EventSource(`/api/demo/stream?${qs}`);
    es.onmessage = (m) => { try { onEvent(JSON.parse(m.data)); } catch { /* ignore */ } };
    es.onerror = () => { onEvent({ event: "error", error: "stream closed" }); es.close(); };
    return () => es.close();
  },
};

export const fmt = {
  pct: (x: number | null | undefined, d = 0) => (x == null || Number.isNaN(x) ? "–" : `${(100 * x).toFixed(d)}%`),
  num: (x: number | null | undefined, d = 2) => (x == null || Number.isNaN(x) ? "–" : x.toFixed(d)),
  mb: (b: number) => `${(b / 2 ** 20).toFixed(1)} MB`,
  s: (x: number | undefined) => (x == null ? "–" : x < 1 ? `${(1000 * x).toFixed(0)} ms` : `${x.toFixed(2)} s`),
};
