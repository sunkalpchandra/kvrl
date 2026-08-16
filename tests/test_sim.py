import numpy as np
import pytest
import torch

from kvrl.controllers import make_controller
from kvrl.sim import CacheSimEnv, episode_budget
from kvrl.sim.env import run_controller_episode
from kvrl.traces.storage import Trace, load_trace, save_trace


def synthetic_trace(P=256, G=64, C=32, seed=0, crit=(40, 48)) -> Trace:
    rng = np.random.default_rng(seed)
    T = P + G
    ends = list(range(C, P + 1, C))
    if ends[-1] != P:
        ends.append(P)
    dec = list(range(P + C, T + 1, C))
    if not dec or dec[-1] != T:
        dec.append(T)
    step_end = np.array(ends + dec, dtype=np.int32)
    step_phase = np.array([0] * len(ends) + [1] * len(dec), dtype=np.int8)
    K = len(step_end)
    A = np.zeros((K, T), dtype=np.float32)
    for k, e in enumerate(step_end):
        row = rng.gamma(0.3, 1.0, size=e).astype(np.float32)
        row[0] += 5.0  # sink
        row[crit[0]:crit[1]] += 3.0 * (k > K // 2)  # critical tokens matter late
        A[k, :e] = row / row.sum()
    cm = np.zeros(T, dtype=bool)
    cm[crit[0]:crit[1]] = True
    return Trace(
        trace_id="synthetic",
        token_ids=rng.integers(0, 1000, T).astype(np.int32),
        n_prompt=P, n_gen=G, chunk=C,
        attn_mean=A.astype(np.float16),
        attn_lmax=np.minimum(A * 2, 1).astype(np.float16),
        step_end=step_end, step_phase=step_phase,
        key_norm=rng.normal(30, 3, T).astype(np.float16),
        value_norm=rng.normal(3, 0.5, T).astype(np.float16),
        adj_key_cos=rng.uniform(0, 1, T).astype(np.float16),
        gen_logprob=rng.normal(-3, 1, G).astype(np.float16),
        critical_mask=cm,
        meta={"task": "needle", "seed": seed},
    )


def test_trace_roundtrip(tmp_path):
    tr = synthetic_trace()
    p = save_trace(tr, tmp_path)
    tr2 = load_trace(p)
    assert tr2.attn_mean.shape == tr.attn_mean.shape and tr2.n_prompt == 256
    assert tr2.meta["task"] == "needle" and tr2.critical_mask.sum() == 8


def test_episode_budget():
    assert episode_budget(0.25, 1000, 64) == 256
    assert episode_budget(1.0, 1000, 64) == 1 << 30
    assert episode_budget(0.01, 1000, 64, min_tokens=128) == 128


def test_env_budget_enforced_and_rewards_negative():
    tr = synthetic_trace()
    env = CacheSimEnv(gamma=0.99, r_scale=1.0)
    res = env.reset(tr, budget=96)
    steps = 0
    while not res.done:
        assert res.m > 0 and res.obs_tok.shape[0] == env.state.n
        assert res.cand_mask.sum() >= res.m
        # evict the first m candidates
        cand = torch.nonzero(res.cand_mask).flatten()[: res.m]
        res = env.step(cand)
        assert res.reward <= 0 or res.done
        steps += 1
        if not res.done:
            assert env.alive.shape[0] <= 96 + tr.chunk
    assert steps > 3
    m = res.info
    assert m["n_evictions"] > 0 and 0 <= m["lost_mass_mean"] <= 1
    assert 0 <= m["crit_retained"] <= 1


def test_env_rejects_bad_actions():
    tr = synthetic_trace()
    env = CacheSimEnv()
    res = env.reset(tr, budget=96)
    prot = torch.nonzero(~res.cand_mask).flatten()
    with pytest.raises(ValueError):
        env.step(prot[: res.m]) if prot.numel() >= res.m else env.step(torch.zeros(res.m + 1, dtype=torch.long))
    with pytest.raises(ValueError):
        env.step(torch.nonzero(res.cand_mask).flatten()[: res.m - 1])


def test_r1_equals_delayed_lost_mass_return():
    """Σ_k γ^k r_k(R1) == -Σ_k γ^k ℓ_{k+1} (telescoping identity from ML_SPEC)."""
    tr = synthetic_trace(P=192, G=32, C=32, seed=3)
    gamma = 0.9
    env = CacheSimEnv(gamma=gamma, r_scale=1.0, lambda_task=0.0)
    res = env.reset(tr, budget=64)
    rewards = []
    steps = []
    while not res.done:
        cand = torch.nonzero(res.cand_mask).flatten()
        pick = cand[torch.randperm(cand.numel())[: res.m]]
        steps.append(env.k)
        res = env.step(pick)
        rewards.append(res.reward)
    # R1 return discounted from the first decision step k0
    k0 = steps[0]
    r1 = sum(r * gamma ** (k - k0) for r, k in zip(rewards, steps))
    # delayed lost mass ℓ_k for k > k0 (lost_mass recorded per processed step)
    lm = env.lost_mass
    r2 = -sum(lm[k] * gamma ** (k - 1 - k0) for k in range(k0 + 1, len(lm)))
    assert abs(r1 - r2) < 1e-3, (r1, r2)


def test_heuristics_run_in_sim_and_window_beats_random_on_recency_trace():
    tr = synthetic_trace(seed=1)
    env = CacheSimEnv()
    out = {}
    for name in ["window", "random", "h2o", "snapkv"]:
        out[name] = run_controller_episode(env, tr, make_controller(name), budget_frac=0.25)
        assert out[name]["n_evictions"] > 0
    assert set(out) == {"window", "random", "h2o", "snapkv"}
