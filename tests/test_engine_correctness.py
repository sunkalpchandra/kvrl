"""Numerical correctness of the real inference path (tiny random model, fp32, CPU)."""

import pytest
import torch

from kvrl.cache.reference import MaskedReference
from kvrl.controllers import make_controller
from kvrl.engine import InferenceEngine, budget_from_fraction
from kvrl.models.attention import CTX
from kvrl.models.hf_model import load_model


@pytest.fixture(scope="module")
def tiny():
    return load_model("tiny-random", device="cpu")


def _prompt(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 1000, (n,), generator=g)


def test_full_cache_matches_hf_greedy(tiny):
    ids = _prompt(150)
    eng = InferenceEngine(tiny, chunk_size=32, decide_every=16)
    res = eng.run(ids, make_controller("full"), budget=1 << 30, max_new_tokens=12, stop_on_eos=False)
    ref = tiny.greedy_reference(ids, 12)[0].tolist()
    assert res.generated_ids == ref
    assert res.n_evicted_total == 0 and res.final_cache_len == 150 + 11


def test_chunked_prefill_matches_oneshot(tiny):
    ids = _prompt(100)
    c1 = tiny.new_cache()
    l1 = tiny.forward_chunk(ids, torch.arange(100), c1)
    c2 = tiny.new_cache()
    pos = 0
    for s in range(0, 100, 16):
        ch = ids[s : s + 16]
        l2 = tiny.forward_chunk(ch, torch.arange(pos, pos + ch.numel()), c2)
        pos += ch.numel()
    assert torch.allclose(l1, l2, atol=1e-5)


@pytest.mark.parametrize("ctrl_name", ["window", "random", "h2o", "snapkv", "keynorm"])
def test_physical_eviction_matches_masked_reference(tiny, ctrl_name):
    """Replay the engine's exact eviction schedule with a masked full cache; log-probs must match."""
    ids = _prompt(160, seed=1)
    eng = InferenceEngine(tiny, chunk_size=32, decide_every=8)
    ctrl = make_controller(ctrl_name)
    budget = 64
    res = eng.run(ids, ctrl, budget=budget, max_new_tokens=24, stop_on_eos=False)
    assert res.n_evicted_total > 0
    assert res.peak_cache_len <= budget + 32
    # replay with masking
    ref = MaskedReference(tiny)
    sched = {d.ctx_len: d.evicted_positions for d in res.decisions}
    pos = 0
    logits = None
    for s in range(0, 160, 32):
        ch = ids[s : s + 32]
        logits = ref.forward_chunk(ch, torch.arange(pos, pos + ch.numel()))
        pos += ch.numel()
        if sched.get(pos):
            ref.evict_positions(torch.tensor(sched[pos]))
    gen = res.generated_ids
    lps = []
    for t, tok in enumerate(gen):
        lps.append(torch.log_softmax(logits[0, -1].float(), -1)[tok].item())
        if t == len(gen) - 1:
            break
        logits = ref.forward_chunk(torch.tensor([tok]), torch.tensor([pos]))
        pos += 1
        if sched.get(pos):
            ref.evict_positions(torch.tensor(sched[pos]))
    assert len(lps) == len(res.token_logprobs)
    assert max(abs(a - b) for a, b in zip(lps, res.token_logprobs)) < 1e-4


def test_position_ids_matter_after_compaction(tiny):
    """Guard: if HF ever stops honouring position_ids the compaction path silently breaks."""
    ids = _prompt(64)
    cache = tiny.new_cache()
    tiny.forward_chunk(ids, torch.arange(64), cache)
    keep = torch.arange(0, 64, 2)  # evict every other token
    from kvrl.cache.compact import compact_cache

    compact_cache(cache, keep)
    nxt = torch.tensor([5, 6, 7])
    import copy

    c_a = copy.deepcopy(cache)
    c_b = copy.deepcopy(cache)
    la = tiny.forward_chunk(nxt, torch.arange(64, 67), c_a)  # correct absolute positions
    lb = tiny.forward_chunk(nxt, torch.arange(32, 35), c_b)  # wrong: cache-length positions
    assert not torch.allclose(la, lb, atol=1e-4)


def test_stats_mass_conservation(tiny):
    ids = _prompt(96)
    tiny.set_stats(True, qblock=16)
    tiny.stats.reset()
    cache = tiny.new_cache()
    tiny.forward_chunk(ids[:64], torch.arange(64), cache)
    tiny.forward_chunk(ids[64:], torch.arange(64, 96), cache)
    n_layers, n_heads = tiny.info.n_layers, tiny.info.n_heads
    total = tiny.stats.mass[:, :96].sum(dim=1)
    # each (layer, head, query) row of the softmax sums to 1 -> mass per layer = heads * queries
    assert torch.allclose(total, torch.full((n_layers,), float(n_heads * 96)), atol=1e-3)
    frac = tiny.stats.normalized(96)
    assert torch.allclose(frac.sum(dim=1), torch.ones(n_layers), atol=1e-4)
    # stats on must not change outputs
    tiny.set_stats(False)
    c2 = tiny.new_cache()
    l_off = tiny.forward_chunk(ids, torch.arange(96), c2)
    tiny.set_stats(True)
    tiny.stats.reset()
    c3 = tiny.new_cache()
    l_on = tiny.forward_chunk(ids, torch.arange(96), c3)
    tiny.set_stats(False)
    assert torch.equal(l_off, l_on)
    assert CTX.calls > 0


def test_budget_from_fraction():
    assert budget_from_fraction(1.0, 8000) == 1 << 30
    assert budget_from_fraction(0.25, 8000, chunk=64) == 2048
    assert budget_from_fraction(0.01, 8000, chunk=64, min_tokens=128) == 128
