import math

from kvrl.eval.metrics import (
    bootstrap_ci,
    lcs_length,
    paired_difference,
    prefix_agreement,
    rouge_l,
    summarize,
    token_agreement,
)


def test_lcs_and_rouge():
    assert lcs_length("abcde", "ace") == 3
    assert lcs_length([], [1]) == 0
    assert rouge_l([1, 2, 3], [1, 2, 3]) == 1.0
    assert rouge_l([1, 2, 3], [4, 5]) == 0.0
    assert 0 < rouge_l([1, 2, 9], [1, 2, 3]) < 1


def test_agreements():
    assert prefix_agreement([1, 2, 3], [1, 2, 4]) == 2 / 3
    assert token_agreement([1, 0, 3], [1, 2, 3]) == 2 / 3
    assert prefix_agreement([], []) == 1.0


def test_bootstrap_and_paired():
    m, lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0], n_boot=500)
    assert lo <= m <= hi and abs(m - 2.5) < 1e-9
    assert math.isnan(bootstrap_ci([])[0])
    res = paired_difference([1, 1, 1, 1, 1], [0, 0, 0, 0, 0], n_boot=200)
    assert res["mean_diff"] == 1.0 and res["a_better_rate"] == 1.0 and res["significant"]
    res_l = paired_difference([1, 1, 1, 1, 1], [0, 0, 0, 0, 0], n_boot=200, lower_is_better=True)
    assert res_l["a_better_rate"] == 0.0 and res_l["significant"]
    res2 = paired_difference([1, 0, 1, 0], [0, 1, 0, 1], n_boot=200)
    assert not res2["significant"]
    s = summarize([0.5, 0.7])
    assert s["n"] == 2 and abs(s["mean"] - 0.6) < 1e-9
