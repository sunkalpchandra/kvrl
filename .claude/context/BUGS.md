# BUGS  (open + resolved; every resolved major bug gets a regression test)

| id | date | area | summary | status | test |
|----|------|------|---------|--------|------|
| BUG-001 | 2026-08-16 | rl/sampler | PPO loss became NaN after the first update: `-inf` padding of picked-score vectors gives NaN gradients through `logcumsumexp` (all-`-inf` prefix) and `0 × inf` in the RB entropy for mixed-m minibatches (first eviction step of an episode has m < 64). Fixed with finite padding + gradient-free `where`. | resolved | tests/test_sampler.py::test_padded_batch_gradients_are_finite |
