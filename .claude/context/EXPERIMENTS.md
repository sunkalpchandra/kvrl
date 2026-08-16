# EXPERIMENTS  (every completed run: config, commit, seed, key numbers, verdict)

Rules: numbers only from `runs/<id>/`; label `sim` vs `real`; record failures too.

| id | date | phase | what | commit | key result | verdict |
|----|------|-------|------|--------|------------|---------|
| E-000 | 2026-08-16 | P0 | dtype/device micro-bench (scratch, 2K ctx, Qwen2.5-0.5B) | 27fe04b | real: MPS fp16 prefill 1751 tok/s, decode 23.4 ms/tok; bf16 668 tok/s / 49.6 ms; fp32 400 / 164; CPU fp32 159 / 211 | fp16 default on MPS (D-004) |
| E-001 | 2026-08-16 | P1 | attention micro-bench (scratch) | (perf commit) | real: enable_gqa SDPA vs repeat_kv: q=1 kv=8K 0.15 vs 4.06 ms; q=64 kv=8K 1.25 vs 4.49 ms; grouped stats pass 2-3× cheaper than expanded | adopted |
| E-002 | 2026-08-16 | P4 | PPO smoke (20 train traces as val, 8 updates × 512 steps, lr 1e-3, ent 1e-3) run 20260816-02xx | (rl commits) | sim: rl@0.25 lost_mass_decode 0.082 vs h2o 0.105, snapkv 0.148, window 0.179, random 0.188, oracle 0.054 (2 val traces) | promising; not a result (train==val, tiny) |
