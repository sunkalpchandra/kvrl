# Failure analysis (sim, val traces, budget 25%) — auto-generated

20 traces · checkpoint `checkpoints/ppo_mlp_v1.pt`

| controller | lost mass (decode) | critical retained @question | critical retained (decode) | evict-age mean |
|---|---|---|---|---|
| oracle | 0.0822 | 0.691 | 0.691 | 13.0 |
| rl | 0.1092 | 0.493 | 0.493 | 11.7 |
| regressor | 0.1153 | 0.141 | 0.141 | 14.0 |
| snapkv | 0.1189 | 0.084 | 0.084 | 14.5 |
| h2o | 0.1220 | 0.339 | 0.339 | 10.7 |
| keynorm | 0.1474 | 0.726 | 0.726 | 5.5 |
| window | 0.1502 | 0.026 | 0.026 | 15.9 |
| random | 0.1641 | 0.113 | 0.113 | 13.5 |

## Critical-token retention at question time, by task

| task       |   h2o |   keynorm |   oracle |   random |   regressor |   rl |   snapkv |   window |
|:-----------|------:|----------:|---------:|---------:|------------:|-----:|---------:|---------:|
| dependency |  0.35 |      0.63 |     0.69 |     0.27 |        0.39 | 0.47 |     0.27 |     0.13 |
| kv         |  0.13 |      0.26 |     0.21 |     0.08 |        0.00 | 0.05 |     0.03 |     0.00 |
| multihop   |  0.30 |      0.83 |     0.87 |     0.08 |        0.25 | 0.40 |     0.10 |     0.00 |
| needle     |  0.45 |      0.95 |     0.84 |     0.07 |        0.03 | 0.77 |     0.01 |     0.00 |

## RL feature permutation importance (Δ lost mass when shuffled; sim)

| feature | Δ lost mass |
|---|---|
| key_norm | +0.0165 |
| attn_mean | +0.0047 |
| attn_lastmax_layer | +0.0032 |
| since_hit | +0.0014 |
| hit_rate | +0.0013 |
| attn_max | +0.0011 |
| attn_ema_slow | +0.0010 |
| attn_last | +0.0009 |
| nbr_evicted | +0.0008 |
| attn_ema_fast | +0.0006 |
| pos_log | +0.0002 |
| adj_key_cos | +0.0000 |
| attn_disp | +0.0000 |
| is_generated | +0.0000 |
| chunk_share | -0.0002 |
| age_log | -0.0002 |
| rel_pos | -0.0003 |
| value_norm | -0.0021 |
