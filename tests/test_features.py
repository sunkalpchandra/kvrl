import torch

from kvrl.cache.view import CacheState
from kvrl.features import GLOBAL_FEATURES, TOKEN_FEATURES, FeatureConfig, FeatureState, phi


def _state(n, step, n_new, chunk=16, seed=0):
    g = torch.Generator().manual_seed(seed + step)
    a = torch.rand(n, generator=g)
    a = a / a.sum()
    return CacheState(
        positions=torch.arange(n),
        chunk_id=torch.arange(n) // chunk,
        is_generated=torch.zeros(n, dtype=torch.bool),
        attn_last_mean=a,
        attn_last_max=(a * 1.5).clamp(max=1),
        attn_cum_mean=a,
        attn_cum_max=a,
        k_norm=torch.rand(n, generator=g) + 1,
        v_norm=torch.rand(n, generator=g) + 1,
        adj_cos=torch.rand(n, generator=g),
        n_new=n_new,
        step=step,
        ctx_len=n,
        phase=0,
        n_prompt=128,
        n_generated=0,
        max_new_tokens=32,
    )


def test_phi_length_invariance():
    # uniform attention over n keys maps to the same value for any n
    for n in (64, 1024, 32768):
        v = phi(torch.tensor([1.0 / n]), n)
        assert abs(v.item() - phi(torch.tensor([1.0 / 64]), 64).item()) < 1e-6


def test_feature_shapes_and_ranges():
    fs = FeatureState()
    st = _state(32, step=1, n_new=16)
    tok, glob = fs.update(st, budget=24)
    assert tok.shape == (32, len(TOKEN_FEATURES)) and glob.shape == (len(GLOBAL_FEATURES),)
    assert torch.isfinite(tok).all() and torch.isfinite(glob).all()
    # second update grows with the cache
    st2 = _state(48, step=2, n_new=16)
    tok2, _ = fs.update(st2, budget=24)
    assert tok2.shape[0] == 48
    # attn_mean of a fresh token equals its attn_last
    i = TOKEN_FEATURES.index
    assert torch.allclose(tok2[32:, i("attn_mean")], tok2[32:, i("attn_last")])
    assert torch.allclose(tok2[32:, i("attn_ema_fast")], tok2[32:, i("attn_last")])
    # ages: older tokens have larger age_log
    assert tok2[0, i("age_log")] > tok2[40, i("age_log")]


def test_compact_keeps_alignment_and_nbr_evicted():
    fs = FeatureState()
    st = _state(32, step=1, n_new=16)
    fs.update(st, budget=16)
    keep = torch.cat([torch.arange(0, 8), torch.arange(16, 32)])  # evict slots 8..15 (chunk 0 half)
    fs.compact(keep, chunk_ids_before=st.chunk_id)
    st2 = _state(24 + 16, step=2, n_new=16)
    st2.chunk_id = torch.cat([st.chunk_id[keep], torch.full((16,), 2)])
    st2.positions = torch.cat([st.positions[keep], torch.arange(32, 48)])
    tok, _ = fs.update(st2, budget=16)
    i = TOKEN_FEATURES.index("nbr_evicted")
    assert torch.allclose(tok[:8, i], torch.full((8,), 0.5))  # chunk 0 lost half
    assert torch.allclose(tok[8:24, i], torch.zeros(16))
    assert fs.n == 40


def test_feature_subset_config():
    cfg = FeatureConfig(token_features=["age_log", "attn_mean"])
    fs = FeatureState(cfg)
    tok, _ = fs.update(_state(20, 1, 10), budget=10)
    assert tok.shape == (20, 2)
