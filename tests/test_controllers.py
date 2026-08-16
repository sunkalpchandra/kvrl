import torch

from kvrl.cache.view import CacheState
from kvrl.controllers import HEURISTICS, make_controller, select_keep


def _state(n=100, n_new=10, n_sink=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return CacheState(
        positions=torch.arange(n),
        chunk_id=torch.arange(n) // 10,
        is_generated=torch.zeros(n, dtype=torch.bool),
        attn_last_mean=torch.rand(n, generator=g),
        attn_last_max=torch.rand(n, generator=g),
        attn_cum_mean=torch.rand(n, generator=g),
        attn_cum_max=torch.rand(n, generator=g),
        k_norm=torch.rand(n, generator=g),
        v_norm=torch.rand(n, generator=g),
        adj_cos=torch.rand(n, generator=g),
        n_new=n_new,
        step=9,
        ctx_len=n,
        phase=0,
        n_prompt=n,
        n_generated=0,
        max_new_tokens=16,
        n_sink=n_sink,
    )


def test_select_keep_respects_budget_and_protection():
    st = _state()
    scores = torch.rand(100)
    keep = select_keep(scores, st, budget=40)
    assert keep.numel() == 40
    assert torch.equal(keep, keep.sort().values) and keep.unique().numel() == 40
    prot = st.protected_mask()
    assert prot[keep].sum() == prot.sum()  # all protected retained
    # highest scoring non-protected are kept
    cand = scores.masked_fill(prot, -1)
    top = torch.topk(cand, 40 - int(prot.sum())).indices
    assert set(top.tolist()) <= set(keep.tolist())
    assert select_keep(scores, st, budget=200).numel() == 100


def test_all_heuristics_return_valid_keep_sets():
    st = _state()
    for name in HEURISTICS:
        c = make_controller(name)
        c.reset(episode=0)
        keep = c.decide(st, budget=50)
        assert keep.dtype == torch.long
        if name == "full":
            assert keep.numel() == 100
        else:
            assert keep.numel() == 50, name
            assert keep.min() >= 0 and keep.max() < 100
            assert torch.equal(keep, keep.unique().sort().values)


def test_window_keeps_recent_and_sinks():
    st = _state()
    keep = make_controller("window").decide(st, budget=30)
    kept = set(keep.tolist())
    assert {0, 1, 2, 3} <= kept
    assert set(range(74, 100)) <= kept


def test_h2o_reserves_recent_half():
    st = _state()
    keep = make_controller("h2o", recent_frac=0.5).decide(st, budget=40)
    kept = set(keep.tolist())
    assert set(range(80, 100)) <= kept  # 20 most recent reserved


def test_random_is_seeded():
    st = _state()
    a = make_controller("random", seed=3)
    b = make_controller("random", seed=3)
    a.reset(episode=1)
    b.reset(episode=1)
    assert torch.equal(a.decide(st, 30), b.decide(st, 30))
