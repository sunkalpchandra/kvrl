# Workflow: experiment

1. Read EXPERIMENTS.md + BENCHMARKS.md (avoid re-running known things).
2. Write/choose a YAML config under configs/; give the run an id `<phase>-<slug>-<seed>`.
3. Run via CLI (`python -m kvrl.train|evaluate|benchmark --config ...`); tracker
   writes `runs/<id>/{config.yaml,meta.json,metrics.jsonl,results.parquet}` incl.
   commit hash, device, seed, timings.
4. Record in EXPERIMENTS.md: id, config, commit, key numbers, verdict (incl. failures).
5. If it changes canonical numbers, update BENCHMARKS.md and README results with
   provenance; regenerate plots via scripts/ (never hand-edit numbers).
