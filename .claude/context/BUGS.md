# BUGS  (open + resolved; every resolved major bug gets a regression test)

| id | date | area | summary | status | test |
|----|------|------|---------|--------|------|
| BUG-001 | 2026-08-16 | rl/sampler | PPO loss became NaN after the first update: `-inf` padding of picked-score vectors gives NaN gradients through `logcumsumexp` (all-`-inf` prefix) and `0 × inf` in the RB entropy for mixed-m minibatches (first eviction step of an episode has m < 64). Fixed with finite padding + gradient-free `where`. | resolved | tests/test_sampler.py::test_padded_batch_gradients_are_finite |
| BUG-002 | 2026-08-16 | cache/stats | StatsBuffer never grows when statistics are disabled (full/window controllers), so `normalized(n)` returned [L, 4096] for n > 4096 and `KVCacheView.state()` crashed on 4K+ prompts (found by scripts/eproxy.py on a 4160-token trace). Fixed: `normalized`/`compact` grow on demand. | resolved | tests/test_engine_correctness.py::test_stats_buffer_grows_when_stats_disabled |
