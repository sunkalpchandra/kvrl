import { useEffect, useState } from "react";
import { isStatic } from "./api";
import { Live } from "./pages/Live";
import { ParetoPage } from "./pages/Pareto";
import { Experiments } from "./pages/Experiments";

const PAGES = [["live", "Live demo"], ["pareto", "Pareto frontier"], ["experiments", "Experiments"]] as const;

export default function App() {
  const [page, setPage] = useState<string>(() => location.hash.replace("#", "") || "live");
  const [stat, setStat] = useState(false);
  useEffect(() => { const h = () => setPage(location.hash.replace("#", "") || "live"); addEventListener("hashchange", h); return () => removeEventListener("hashchange", h); }, []);
  useEffect(() => { isStatic().then(setStat); }, []);
  return (
    <div className="mx-auto max-w-6xl px-6 py-6">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4 border-b border-zinc-800 pb-4">
        <div>
          <div className="label">kvrl</div>
          <h1 className="text-xl font-semibold tracking-tight text-zinc-100">Adaptive KV cache</h1>
          <p className="mt-1 max-w-2xl text-sm text-zinc-500">An RL controller decides which tokens a Transformer keeps in its KV cache during long-context inference. Every number here is measured on the real model{stat ? " (static snapshot)" : ""}.</p>
        </div>
        <nav className="flex gap-1">{PAGES.map(([id, name]) => (
          <a key={id} href={`#${id}`} className={`rounded-md px-3 py-1.5 text-sm ${page === id ? "bg-zinc-800 text-zinc-100" : "text-zinc-400 hover:text-zinc-200"}`}>{name}</a>))}</nav>
      </header>
      {page === "live" && <Live isStatic={stat} />}
      {page === "pareto" && <ParetoPage />}
      {page === "experiments" && <Experiments />}
      <footer className="mt-10 border-t border-zinc-800 pt-3 text-xs text-zinc-600">github.com/sunkalpchandra/kvrl · numbers are never hard-coded: they come from runs/ or a live inference in this session</footer>
    </div>
  );
}
