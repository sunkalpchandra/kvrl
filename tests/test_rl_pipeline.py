"""RL pipeline on a synthetic trace: rollout → PPO update → checkpoint → RLController in sim & engine."""

import torch

from kvrl.controllers import make_controller
from kvrl.controllers.learned import RLController, load_policy_checkpoint, save_policy_checkpoint
from kvrl.engine import InferenceEngine
from kvrl.features import GLOBAL_FEATURES, TOKEN_FEATURES, FeatureConfig
from kvrl.models.hf_model import load_model
from kvrl.rl.policy import ScorePolicy, ValueNet
from kvrl.rl.ppo import PPO, PPOConfig, RolloutBuffer, Transition
from kvrl.sim.env import CacheSimEnv, run_controller_episode
from tests.test_sim import synthetic_trace


def _rollout(env, algo, trace, gen, buf, budget=96):
    res = env.reset(trace, budget=budget)
    while not res.done:
        priv = env.privileged()
        ev, lp, v, _ = algo.act(res.obs_tok, res.obs_glob, res.cand_mask, res.m, priv=priv, generator=gen)
        nxt = env.step(ev)
        buf.add(Transition(res.obs_tok.half(), res.obs_glob, res.cand_mask, ev, lp, v, nxt.reward, priv, 0))
        res = nxt
    buf.end_episode(0.0)
    return res.info


def test_ppo_update_runs_and_changes_policy():
    torch.manual_seed(0)
    fcfg = FeatureConfig()
    env = CacheSimEnv(feature_cfg=fcfg, r_scale=0.1)
    policy = ScorePolicy(len(TOKEN_FEATURES), len(GLOBAL_FEATURES))
    value = ValueNet(len(TOKEN_FEATURES), len(GLOBAL_FEATURES))
    algo = PPO(policy, value, PPOConfig(minibatch=16, epochs=2))
    gen = torch.Generator().manual_seed(0)
    buf = RolloutBuffer()
    before = [p.detach().clone() for p in policy.parameters()]
    for seed in range(3):
        _rollout(env, algo, synthetic_trace(seed=seed), gen, buf)
    assert len(buf) > 10
    stats = algo.update(buf)
    assert all(torch.isfinite(torch.tensor(v)).all() for v in stats.values() if isinstance(v, float))
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters()))
    assert stats["approx_kl"] >= 0 and stats["entropy"] > 0


def test_checkpoint_roundtrip_and_controller_in_sim_and_engine(tmp_path):
    fcfg = FeatureConfig(token_features=["age_log", "rel_pos", "attn_last", "attn_mean"])
    policy = ScorePolicy(4, len(GLOBAL_FEATURES))
    p = save_policy_checkpoint(tmp_path / "pol.pt", policy, fcfg, kind="rl", meta={"note": "test"})
    policy2, fcfg2, ck = load_policy_checkpoint(p)
    assert ck["kind"] == "rl" and fcfg2.token_features == fcfg.token_features
    for a, b in zip(policy.parameters(), policy2.parameters()):
        assert torch.equal(a, b)
    ctrl = RLController(policy2, fcfg2)
    env = CacheSimEnv(feature_cfg=fcfg2)
    info = run_controller_episode(env, synthetic_trace(seed=4), ctrl, budget_frac=0.3)
    assert info["n_evictions"] > 0
    # in the real engine (tiny model) with stats on
    tiny = load_model("tiny-random", device="cpu")
    eng = InferenceEngine(tiny, chunk_size=32, decide_every=16)
    ids = torch.randint(0, 1000, (200,))
    ctrl2 = make_controller("rl", checkpoint=p)
    res = eng.run(ids, ctrl2, budget=80, max_new_tokens=20, record_importance=True, stop_on_eos=False)
    assert res.n_evicted_total > 0 and res.stats_enabled and res.peak_cache_len <= 80 + 32
    assert all(len(d.importance) == d.n_after for d in res.decisions if d.importance is not None)
    # deterministic: same run twice gives the same evictions
    res2 = eng.run(ids, make_controller("rl", checkpoint=p), budget=80, max_new_tokens=20, stop_on_eos=False)
    assert [d.evicted_positions for d in res.decisions] == [d.evicted_positions for d in res2.decisions]
