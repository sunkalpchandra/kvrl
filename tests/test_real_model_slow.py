"""Real-model integration tests (need the downloaded model; run with `pytest -m slow`)."""

from pathlib import Path

import pytest
import torch

from kvrl.cache.reference import MaskedReference
from kvrl.controllers import make_controller
from kvrl.engine import InferenceEngine
from kvrl.eval.corpus import Filler, load_corpus
from kvrl.eval.tasks import generate

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def real():
    from kvrl.models.hf_model import load_model

    try:
        return load_model("qwen2.5-0.5b-instruct")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"model unavailable: {e!r}")


@pytest.fixture(scope="module")
def prompt(real):
    filler = Filler(load_corpus(allow_download=False), seed=3)
    inst = generate("needle", 1, 1024, 3, filler, count_tokens=real.count_tokens)[0]
    return real.encode_chat(inst.prompt, inst.system), inst


def test_full_cache_matches_hf_greedy_real(real, prompt):
    ids, _ = prompt
    eng = InferenceEngine(real, chunk_size=64, decide_every=64)
    res = eng.run(ids, make_controller("full"), budget=1 << 30, max_new_tokens=12, stop_on_eos=False)
    ref = real.greedy_reference(ids, 12)[0].tolist()
    assert res.generated_ids == ref


def test_eviction_matches_masking_real(real, prompt):
    ids, _ = prompt
    eng = InferenceEngine(real, chunk_size=64, decide_every=8)
    res = eng.run(ids, make_controller("h2o"), budget=256, max_new_tokens=8, stop_on_eos=False)
    ref = MaskedReference(real)
    sched = {d.ctx_len: d.evicted_positions for d in res.decisions}
    pos = 0
    logits = None
    n = int(ids.numel())
    for s in range(0, n, 64):
        ch = ids[s : s + 64]
        logits = ref.forward_chunk(ch, torch.arange(pos, pos + ch.numel()))
        pos += ch.numel()
        if sched.get(pos):
            ref.evict_positions(torch.tensor(sched[pos]))
    agree = 0
    for t, tok in enumerate(res.generated_ids):
        agree += int(logits[0, -1].argmax().item() == tok)
        if t == len(res.generated_ids) - 1:
            break
        logits = ref.forward_chunk(torch.tensor([tok]), torch.tensor([pos]))
        pos += 1
        if sched.get(pos):
            ref.evict_positions(torch.tensor(sched[pos]))
    # fp16 on MPS: identical argmax expected on the vast majority of steps
    assert agree >= len(res.generated_ids) - 1


@pytest.mark.skipif(not Path("checkpoints/ppo_mlp_v1.pt").exists(), reason="no trained policy")
def test_rl_controller_runs_end_to_end(real, prompt):
    ids, _inst = prompt
    eng = InferenceEngine(real, chunk_size=64, decide_every=64)
    ctrl = make_controller("rl", checkpoint="checkpoints/ppo_mlp_v1.pt")
    res = eng.run(ids, ctrl, budget=256, max_new_tokens=8, record_importance=True)
    assert res.n_evicted_total > 0 and res.peak_cache_len <= 256 + 64
    assert all(len(d.importance or []) == d.n_after for d in res.decisions)
    assert res.timings["controller_s"] < res.timings["model_s"]  # overhead sanity
