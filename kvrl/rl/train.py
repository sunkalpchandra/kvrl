"""PPO training in the cache simulator (Env A) with periodic validation vs heuristics."""

from __future__ import annotations

import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from kvrl.controllers import make_controller
from kvrl.controllers.learned import OracleController, RLController, save_policy_checkpoint
from kvrl.features import GLOBAL_FEATURES, TOKEN_FEATURES, FeatureConfig
from kvrl.sim.env import CacheSimEnv, run_controller_episode
from kvrl.traces.storage import Trace, load_trace, trace_index
from kvrl.utils.seed import seed_everything
from kvrl.utils.tracker import Run

from .policy import ScorePolicy, ValueNet
from .ppo import PPO, PPOConfig, RolloutBuffer, Transition


def load_traces(
    directory: str | Path,
    limit: int | None = None,
    tasks: list[str] | None = None,
    max_prompt: int | None = None,
) -> list[Trace]:
    df = trace_index(directory)
    if len(df) == 0:
        return []
    if tasks:
        df = df[df["task"].isin(tasks)]
    if max_prompt:
        df = df[df["n_prompt"] <= max_prompt]
    paths = list(df["path"])
    if limit:
        paths = paths[:limit]
    return [load_trace(p) for p in paths]


def feature_norm_constants(traces: list[Trace]) -> dict:
    k = np.concatenate([t.key_norm.astype(np.float32) for t in traces])
    v = np.concatenate([t.value_norm.astype(np.float32) for t in traces])
    return {
        "k_norm_mean": float(k.mean()),
        "k_norm_std": float(k.std() + 1e-6),
        "v_norm_mean": float(v.mean()),
        "v_norm_std": float(v.std() + 1e-6),
    }


def calibrate_r_scale(
    env: CacheSimEnv, traces: list[Trace], budget_frac: float, n_episodes: int, seed: int = 0
) -> float:
    """mean |r_k| under a random policy (ML_SPEC §4.2); makes rewards O(1)."""
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    old = env.r_scale
    env.r_scale = 1.0
    mags = []
    for _ in range(n_episodes):
        tr = rng.choice(traces)
        res = env.reset(tr, budget_frac=budget_frac)
        while not res.done:
            cand = torch.nonzero(res.cand_mask).flatten()
            pick = cand[torch.randperm(cand.numel(), generator=g)[: res.m]]
            res = env.step(pick)
            if not res.done:
                mags.append(abs(res.reward))
    env.r_scale = old
    return float(np.mean(mags)) if mags else 1.0


def evaluate(
    policy: ScorePolicy,
    feature_cfg: FeatureConfig,
    traces: list[Trace],
    budget_fracs: list[float],
    env_kwargs: dict,
    baselines: tuple[str, ...] = ("window", "h2o", "snapkv", "random"),
    with_oracle: bool = True,
    seed: int = 0,
) -> dict:
    """Deterministic policy vs heuristics vs oracle on the given traces (sim metrics)."""
    env = CacheSimEnv(**env_kwargs)
    rows = []
    for tr in traces:
        for bf in budget_fracs:
            ctrls = {"rl": RLController(policy, feature_cfg, deterministic=True)}
            for b in baselines:
                ctrls[b] = make_controller(b, seed=seed) if b == "random" else make_controller(b)
            if with_oracle:
                ctrls["oracle"] = OracleController(env.future_mass)
            for name, c in ctrls.items():
                info = run_controller_episode(env, tr, c, budget_frac=bf)
                rows.append(
                    {
                        "trace_id": tr.trace_id,
                        "task": tr.meta.get("task"),
                        "budget_frac": bf,
                        "controller": name,
                        **{k: v for k, v in info.items() if isinstance(v, int | float)},
                    }
                )
    import pandas as pd

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["controller", "budget_frac"])[
            ["lost_mass_decode", "lost_mass_mean", "crit_retained", "total_reward"]
        ]
        .mean()
        .reset_index()
    )
    return {"rows": rows, "summary": summary.to_dict(orient="records")}


def train(cfg: dict, run: Run, log=print) -> dict:
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    rng = random.Random(seed)
    tcfg, rl, sim = cfg["training"], cfg["rl"], cfg["sim"]
    device = cfg.get("device", "cpu")
    # ---------------------------------------------------------------- data
    train_traces = load_traces(
        cfg["data"]["train_dir"],
        limit=cfg["data"].get("limit"),
        tasks=cfg["data"].get("tasks"),
        max_prompt=cfg["data"].get("max_prompt"),
    )
    val_traces = load_traces(
        cfg["data"]["val_dir"],
        limit=cfg["data"].get("val_limit"),
        tasks=cfg["data"].get("tasks"),
        max_prompt=cfg["data"].get("max_prompt"),
    )
    if not train_traces:
        raise RuntimeError(f"no traces in {cfg['data']['train_dir']} — run kvrl.collect first")
    log(f"[train] {len(train_traces)} train traces, {len(val_traces)} val traces")
    consts = feature_norm_constants(train_traces)
    init_from = rl.get("init_from")
    if init_from:
        # warm start from a checkpoint (e.g. the supervised regressor): reuse ITS feature
        # config so normalisation constants match the pretrained weights
        from kvrl.controllers.learned import load_policy_checkpoint

        _p, _f, _ck = load_policy_checkpoint(init_from)
        consts = {
            "k_norm_mean": _f.k_norm_mean,
            "k_norm_std": _f.k_norm_std,
            "v_norm_mean": _f.v_norm_mean,
            "v_norm_std": _f.v_norm_std,
        }
        log(f"[train] warm start from {init_from} ({_ck.get('kind')})")
    fcfg = FeatureConfig(
        **consts,
        token_features=cfg.get("features", {}).get("token", list(TOKEN_FEATURES)),
        ema_fast=cfg.get("features", {}).get("ema_fast", 0.5),
        ema_slow=cfg.get("features", {}).get("ema_slow", 0.9),
    )
    env_kwargs = dict(
        gamma=float(rl["gamma"]),
        lambda_task=float(sim.get("lambda_task", 1.0)),
        n_sink=int(sim.get("n_sink", 4)),
        feature_cfg=fcfg,
        use_layer_max_reward=bool(sim.get("layer_max_reward", False)),
    )
    env = CacheSimEnv(**env_kwargs)
    budget_fracs = [float(b) for b in sim["budget_fracs"]]
    r_scale = sim.get("r_scale")
    if r_scale is None:
        r_scale = calibrate_r_scale(
            env,
            train_traces,
            budget_frac=budget_fracs[0],
            n_episodes=int(sim.get("calib_episodes", 8)),
            seed=seed,
        )
        log(f"[train] calibrated r_scale = {r_scale:.4f}")
    env.r_scale = float(r_scale)
    run.save_json("feature_config.json", asdict(fcfg))
    # ---------------------------------------------------------------- nets
    n_tok, n_glob = len(fcfg.token_features), len(GLOBAL_FEATURES)
    policy = ScorePolicy(
        n_tok, n_glob, hidden=int(rl.get("hidden", 128)), arch=rl.get("arch", "mlp")
    )
    value = ValueNet(n_tok, n_glob, n_priv=3, hidden=int(rl.get("hidden", 128)))
    ppo_cfg = PPOConfig(
        lr=float(rl["lr"]),
        lr_end=float(rl.get("lr_end", rl["lr"] / 10)),
        clip=float(rl.get("clip", 0.2)),
        entropy_coef=float(rl.get("entropy_coef", 0.01)),
        value_coef=float(rl.get("value_coef", 0.5)),
        epochs=int(rl.get("epochs", 4)),
        minibatch=int(rl.get("minibatch", 256)),
        target_kl=float(rl.get("target_kl", 0.02)),
        ratio_mode=rl.get("ratio_mode", "per_slot"),
    )
    if init_from:
        policy.load_state_dict(_p.state_dict())
    algo = PPO(policy, value, ppo_cfg, device=device)
    use_priv = bool(rl.get("privileged_critic", True))
    n_value_params = sum(p.numel() for p in value.parameters())
    log(f"[train] policy {policy.arch} params={policy.n_params()} value params={n_value_params}")
    # ---------------------------------------------------------------- loop
    steps_per_update = int(tcfg["steps_per_update"])
    total_steps = int(tcfg["total_steps"])
    eval_every = int(tcfg.get("eval_every_updates", 10))
    gen = torch.Generator().manual_seed(seed)
    buf = RolloutBuffer(gamma=float(rl["gamma"]), lam=float(rl.get("gae_lambda", 0.95)))
    steps_done, update, episode = 0, 0, 0
    best_val = float("inf")
    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_name = cfg.get("checkpoint_name", "policy")
    ep_returns: list[float] = []
    ep_lost: list[float] = []
    t_start = time.time()
    history = []
    while steps_done < total_steps:
        # -------- rollout
        buf.clear()
        t_roll = time.time()
        while len(buf) < steps_per_update:
            tr = rng.choice(train_traces)
            bf = rng.choice(budget_fracs)
            res = env.reset(tr, budget_frac=bf)
            ep_reward = 0.0
            while not res.done:
                priv = env.privileged() if use_priv else torch.zeros(3)
                ev, lp, v, _ = algo.act(
                    res.obs_tok, res.obs_glob, res.cand_mask, res.m, priv=priv, generator=gen
                )
                nxt = env.step(ev)
                buf.add(
                    Transition(
                        res.obs_tok.half(),
                        res.obs_glob,
                        res.cand_mask,
                        ev,
                        lp,
                        v,
                        nxt.reward,
                        priv,
                        episode,
                    )
                )
                ep_reward += nxt.reward
                res = nxt
            buf.end_episode(0.0)
            episode += 1
            ep_returns.append(res.info.get("total_reward", ep_reward))
            ep_lost.append(res.info.get("lost_mass_decode", float("nan")))
        roll_s = time.time() - t_roll
        # -------- update
        frac = steps_done / total_steps
        lr = algo.set_lr(frac)
        t_upd = time.time()
        stats = algo.update(buf)
        upd_s = time.time() - t_upd
        steps_done += len(buf)
        update += 1
        rec = {
            "update": update,
            "steps": steps_done,
            "episodes": episode,
            "lr": lr,
            "ep_return_mean": float(np.mean(ep_returns[-32:])),
            "ep_lost_mass_decode": float(np.nanmean(ep_lost[-32:])),
            "rollout_s": round(roll_s, 1),
            "update_s": round(upd_s, 1),
            **stats,
        }
        run.log(step=steps_done, **rec)
        history.append(rec)
        log(
            f"[train] upd {update} steps {steps_done} ret {rec['ep_return_mean']:.3f} "
            f"lost_dec {rec['ep_lost_mass_decode']:.4f} kl {stats['approx_kl']:.4f} "
            f"ent {stats['entropy']:.2f} ev {stats['explained_variance']:.2f} "
            f"({roll_s:.0f}s+{upd_s:.0f}s)"
        )
        # -------- eval + checkpoint
        if update % eval_every == 0 or steps_done >= total_steps:
            if val_traces:
                ev_res = evaluate(
                    policy,
                    fcfg,
                    val_traces[: int(tcfg.get("eval_traces", 8))],
                    budget_fracs,
                    env_kwargs | {"r_scale": env.r_scale},
                    seed=seed,
                )
                summ = {(r["controller"], r["budget_frac"]): r for r in ev_res["summary"]}
                rl_lost = float(
                    np.mean(
                        [
                            r["lost_mass_decode"]
                            for r in ev_res["summary"]
                            if r["controller"] == "rl"
                        ]
                    )
                )
                run.log(step=steps_done, eval=ev_res["summary"])
                line = " | ".join(
                    f"{c}@{b}: {summ[(c, b)]['lost_mass_decode']:.4f}" for (c, b) in sorted(summ)
                )
                log(f"[eval ] {line}")
                if rl_lost < best_val:
                    best_val = rl_lost
                    save_policy_checkpoint(
                        ckpt_dir / f"{ckpt_name}.pt",
                        policy,
                        fcfg,
                        kind="rl",
                        meta={
                            "run_id": run.run_id,
                            "update": update,
                            "steps": steps_done,
                            "val_lost_mass_decode": rl_lost,
                            "r_scale": env.r_scale,
                            "config": cfg,
                        },
                    )
                    run.save_json("best_eval.json", ev_res["summary"])
            save_policy_checkpoint(
                ckpt_dir / f"{ckpt_name}_last.pt",
                policy,
                fcfg,
                kind="rl",
                meta={
                    "run_id": run.run_id,
                    "update": update,
                    "steps": steps_done,
                    "r_scale": env.r_scale,
                    "config": cfg,
                },
            )
    return {
        "updates": update,
        "steps": steps_done,
        "episodes": episode,
        "best_val_lost_mass_decode": best_val,
        "train_seconds": round(time.time() - t_start, 1),
        "r_scale": env.r_scale,
        "policy_params": policy.n_params(),
        "history_tail": history[-3:],
    }
