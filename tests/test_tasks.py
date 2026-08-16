from pathlib import Path

from kvrl.eval.corpus import Filler, load_corpus, template_filler
from kvrl.eval.tasks import (
    TASKS,
    approx_count_tokens,
    generate,
    is_correct,
    normalize_answer,
)


def _filler():
    corpus = load_corpus(Path("data/corpus"), allow_download=False)
    return Filler(corpus, seed=0)


def test_template_filler_deterministic():
    assert template_filler(500, seed=1) == template_filler(500, seed=1)
    assert len(template_filler(500, seed=1)) >= 500


def test_all_tasks_generate_and_have_critical_spans():
    f = _filler()
    for name in TASKS:
        xs = generate(name, 2, 512, seed=1, filler=f)
        assert len(xs) >= 1, name
        for x in xs:
            assert x.task == name
            assert isinstance(x.prompt, str) and len(x.prompt) > 100
            if name != "lm":
                assert x.answers, name
                assert x.critical_spans, name
                for a, b in x.critical_spans:
                    assert 0 <= a < b <= len(x.prompt)
                # answers should not appear outside critical spans for retrieval tasks
                if name in ("needle", "kv"):
                    ans = x.answers[0]
                    inside = any(ans in x.prompt[a:b] for a, b in x.critical_spans)
                    assert inside
            else:
                assert x.continuation and len(x.continuation) > 20


def test_lengths_track_target():
    f = _filler()
    for name in ("needle", "multihop", "dependency"):
        for target in (512, 2048):
            x = generate(name, 1, target, seed=3, filler=f)[0]
            n = approx_count_tokens(x.prompt)
            assert abs(n - target) / target < 0.08, (name, target, n)


def test_needle_depth_control():
    f = _filler()
    xs = generate("needle", 5, 2048, seed=0, filler=f, depths=[0.05, 0.5, 0.95])
    rel = [x.critical_spans[0][0] / len(x.prompt) for x in xs[:3]]
    assert rel[0] < 0.2 and 0.3 < rel[1] < 0.7 and rel[2] > 0.8


def test_dependency_answer_is_consistent():
    f = _filler()
    x = generate("dependency", 1, 1024, seed=7, filler=f, chain_len=3)[0]
    # recompute from the critical sentences
    sents = [x.prompt[a:b] for a, b in x.critical_spans]
    val = None
    env = {}
    for s in sents:
        w = s.rstrip(".").split()
        if len(w) == 4:  # Set X to N.
            env[w[1]] = int(w[3])
        else:  # Set X to Y plus/minus D.
            env[w[1]] = env[w[3]] + int(w[5]) * (1 if w[4] == "plus" else -1)
        val = env[w[1]]
    assert str(val) == x.answers[0]


def test_generation_is_deterministic():
    f1, f2 = _filler(), _filler()
    a = generate("multihop", 2, 700, seed=11, filler=f1)
    b = generate("multihop", 2, 700, seed=11, filler=f2)
    assert [x.prompt for x in a] == [x.prompt for x in b]


def test_answer_matching():
    assert normalize_answer("  The Code: '1234-Alpha'. ") == "the code 1234-alpha"
    assert is_correct("1234-alpha", ["1234-alpha"])
    assert is_correct("The value is 1234-alpha\nmore text", ["1234-alpha"])
    assert not is_correct("1234-bravo", ["1234-alpha"])
    assert not is_correct("", ["x"])
