# Ablations (sim, val traces, 20K decision steps each, warm-start protocol, seed 0) — auto-generated

Feature variants warm-start from a regressor trained on the same feature subset; reward variants from the full regressor. Metrics: deterministic policy on val traces at 25 % budget — lost attention mass during decode (lower is better; H2O = 0.1116, SnapKV = 0.1058, oracle = 0.0777) and fraction of task-critical tokens retained (higher is better; H2O = 0.243, keynorm ≈ 0.73).

| group | variant | lost mass @25% | critical retained @25% | run |
|---|---|---|---|---|
| features | full | 0.1086 | 0.390 | `20260816-144010-train-09bfe7487b` |
| features | age_only | 0.1315 | 0.170 | `20260816-144616-train-3c5ad3a58c` |
| features | no_attention | 0.1206 | 0.329 | `20260816-145159-train-a5a7d5b8f1` |
| features | no_history | 0.1102 | 0.345 | `20260816-145747-train-1d06f67c09` |
| features | no_norms | 0.1150 | 0.329 | `20260816-150354-train-8405fbdfd2` |
| features | attention_only | 0.1106 | 0.319 | `20260816-151005-train-b0fbfcf4e4` |
| reward | layer_max_crit | 0.1086 | 0.390 | `20260816-151547-train-43ff8b6861` |
| reward | layer_mean | 0.1093 | 0.347 | `20260816-152145-train-5bfbc9822a` |
| reward | no_crit_penalty | 0.1042 | 0.287 | `20260816-152740-train-d677f53658` |
| reward | no_task_term | 0.1087 | 0.390 | `20260816-153256-train-e2e761dbad` |
| reward | no_privileged_critic | 0.1083 | 0.360 | `20260816-153847-train-89e0ed9000` |
| reward | shared_advantage | 0.1051 | 0.276 | `20260816-154436-train-6e6e8e2992` |

Reading: attention statistics carry most of the lost-mass signal (age-only and no-attention fall below H2O); key/value norms add a little; history features (EMAs/hits) barely matter at this training length. The critical-eviction penalty and per-slot credit assignment trade ~0.004 lost mass for critical-token retention (see the column), which is what they were designed to do — but E-017 shows that this sim gain did not translate to real accuracy.
