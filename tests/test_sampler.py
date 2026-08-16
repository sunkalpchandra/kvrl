import itertools
import math

import torch

from kvrl.rl.sampler import (
    deterministic_evict,
    entropy,
    exact_set_log_prob_bruteforce,
    log_prob,
    sample_evict,
)


def _all_ordered_logps(scores, cand, m):
    n = scores.numel()
    items = [i for i in range(n) if cand[i]]
    out = {}
    for perm in itertools.permutations(items, m):
        lp, valid = log_prob(scores, cand, torch.tensor(perm))
        out[perm] = lp[valid].sum().item()
    return out


def test_ordered_probabilities_sum_to_one():
    torch.manual_seed(0)
    scores = torch.randn(6)
    cand = torch.tensor([True, True, False, True, True, True])
    for m in (1, 2, 3):
        lps = _all_ordered_logps(scores, cand, m)
        assert abs(sum(math.exp(v) for v in lps.values()) - 1.0) < 1e-6, m


def test_m1_is_softmax():
    scores = torch.tensor([0.5, -1.0, 2.0, 0.0])
    cand = torch.ones(4, dtype=torch.bool)
    lp, _ = log_prob(scores, cand, torch.tensor([2]))
    assert abs(lp[0, 0].item() - torch.log_softmax(scores, 0)[2].item()) < 1e-6


def test_gumbel_topk_frequencies_match_plackett_luce():
    torch.manual_seed(1)
    scores = torch.tensor([1.0, 0.0, -0.5, 0.7])
    cand = torch.ones(4, dtype=torch.bool)
    lps = _all_ordered_logps(scores, cand, 2)
    g = torch.Generator().manual_seed(123)
    counts = dict.fromkeys(lps, 0)
    N = 40000
    idx = sample_evict(scores[None].expand(N, -1).contiguous(), cand[None].expand(N, -1).contiguous(),
                       torch.full((N,), 2), generator=g)
    for row in idx.tolist():
        counts[tuple(row)] += 1
    for perm, lp in lps.items():
        p = math.exp(lp)
        f = counts[perm] / N
        # 4 sigma tolerance
        assert abs(f - p) < 4 * math.sqrt(p * (1 - p) / N) + 1e-3, (perm, p, f)


def test_set_probability_bruteforce_matches():
    torch.manual_seed(2)
    scores = torch.randn(5)
    cand = torch.ones(5, dtype=torch.bool)
    lp_set = exact_set_log_prob_bruteforce(scores, cand, {0, 3})
    # sum over the two orderings via log_prob directly
    a, _ = log_prob(scores, cand, torch.tensor([0, 3]))
    b, _ = log_prob(scores, cand, torch.tensor([3, 0]))
    assert abs(math.log(math.exp(a.sum()) + math.exp(b.sum())) - lp_set) < 1e-6


def test_entropy_matches_exact_conditional_entropies():
    torch.manual_seed(3)
    scores = torch.randn(6)
    cand = torch.tensor([True, True, True, False, True, True])
    ev = torch.tensor([4, 1, 0])
    h, _valid = entropy(scores, cand, ev)
    # exact: entropy of softmax over remaining set at each slot
    remaining = [i for i in range(6) if cand[i]]
    for j, i in enumerate(ev.tolist()):
        logits = scores[remaining]
        pr = torch.softmax(logits, 0)
        H = -(pr * torch.log(pr)).sum().item()
        assert abs(h[0, j].item() - H) < 1e-5, (j, h[0, j].item(), H)
        remaining.remove(i)


def test_batched_padding_equals_single():
    torch.manual_seed(4)
    s1, s2 = torch.randn(7), torch.randn(5)
    c1, c2 = torch.ones(7, dtype=torch.bool), torch.tensor([True, False, True, True, True])
    e1, e2 = torch.tensor([2, 5, 0]), torch.tensor([3, 0])
    S = torch.full((2, 7), float("-inf"))
    S[0] = s1
    S[1, :5] = s2
    C = torch.zeros(2, 7, dtype=torch.bool)
    C[0] = c1
    C[1, :5] = c2
    E = torch.full((2, 3), -1, dtype=torch.long)
    E[0] = e1
    E[1, :2] = e2
    lp, _valid = log_prob(S, C, E)
    lp1, _ = log_prob(s1, c1, e1)
    lp2, _ = log_prob(s2, c2, e2)
    assert torch.allclose(lp[0], lp1[0], atol=1e-6)
    assert torch.allclose(lp[1, :2], lp2[0], atol=1e-6) and lp[1, 2] == 0
    h, _ = entropy(S, C, E)
    h2, _ = entropy(s2, c2, e2)
    assert torch.allclose(h[1, :2], h2[0], atol=1e-6)


def test_gradients_flow_and_deterministic_topk():
    scores = torch.randn(10, requires_grad=True)
    cand = torch.ones(10, dtype=torch.bool)
    ev = deterministic_evict(scores.detach(), cand, 3)
    assert set(ev.tolist()) == set(torch.topk(scores, 3).indices.tolist())
    lp, _ = log_prob(scores, cand, ev)
    lp.sum().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_masked_tokens_never_sampled():
    scores = torch.zeros(20)
    cand = torch.zeros(20, dtype=torch.bool)
    cand[[3, 7, 11]] = True
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        idx = sample_evict(scores, cand, 2, generator=g)
        assert set(idx.tolist()) <= {3, 7, 11}
