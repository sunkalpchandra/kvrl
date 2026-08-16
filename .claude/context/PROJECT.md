# PROJECT

## One-line goal

Build an end-to-end system in which a reinforcement-learning controller decides,
during long-context Transformer inference, which entries of the KV cache to keep or
evict (later: merge/compress) so that KV memory and inference cost drop while
downstream task quality is preserved — and measure it honestly against strong
heuristic baselines on real hardware.

## Central question

> Can an RL controller learn which information a Transformer should retain, evict,
> merge, or compress during long-context inference in order to reduce KV-cache
> memory and inference cost while preserving downstream performance?

## What "done" means (success criteria, from the master build prompt)

- A real causal Transformer runs locally (HF), K/V tensors are inspected & manipulated.
- Full-cache + multiple heuristic controllers share one interface and work.
- Fast simulator built from recorded traces supports millions of transitions.
- PPO policy trains in the simulator; the trained policy controls the *real* cache.
- Correctness: budget=100% reproduces HF greedy output; eviction == masking reference.
- Real latency + memory measured (MPS here; CUDA path kept working by abstraction).
- Long-context tasks (retrieval, needle, multi-hop, code-context, synthetic dependency,
  long-doc QA/summarization) run automatically with recorded results.
- Interactive dashboard (Pareto frontier, token retention view, decision trace).
- Fresh environment can run `python demo.py`.

## Non-goals (v1)

- Beating published SOTA methods; the aim is a rigorous, working systems project.
- Training/serving models > ~1.5B params on this laptop.
- Distributed / multi-GPU serving.

## Constraints that shape everything

- **8 GB unified memory, no CUDA, ~9 GB free disk.** Model must be ≤ ~1 GB in
  fp16/bf16; contexts up to 8K–16K realistic for real inference; 32K only via
  chunked prefill and careful measurement; traces must be compact (Parquet/npz,
  aggregated statistics, not full attention tensors).
- **Simulator first.** RL is trained on recorded attention/KV statistics, then
  validated in real inference. Real-inference RL fine-tuning is a late phase.
- **Honesty.** Every number traceable to a run; sim vs real always labelled.

## Milestone plan (vertical slice first)

M1 (vertical slice): small model + 8K context + binary KEEP/EVICT + PPO + simulator
+ real inference + sliding-window & H2O baselines + benchmark harness + tests.
Then: learned token representation, hierarchical/chunk decisions, hardware-aware
objective, full benchmark suite, dashboard, optimisation, docs/demo polish.
