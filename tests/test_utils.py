import json

import pytest
import torch

from kvrl.utils.config import (
    config_hash,
    deep_update,
    get_dotted,
    load_config,
    parse_override,
    set_dotted,
)
from kvrl.utils.device import PeakTracker, memory_stats, pick_dtype, resolve_device, synchronize
from kvrl.utils.timing import Samples, Stopwatch
from kvrl.utils.tracker import list_runs, load_run, start_run


def test_config_overrides_and_hash(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("model:\n  name: a\n  dtype: bf16\nrl:\n  lr: 0.001\n")
    child = tmp_path / "child.yaml"
    child.write_text("_base: base.yaml\nrl:\n  gamma: 0.99\n")
    cfg = load_config(child, overrides=["rl.lr=1e-4", "cache.budget=0.25", "flag=true"])
    assert cfg["model"]["name"] == "a"
    assert cfg["rl"]["gamma"] == 0.99
    assert cfg["rl"]["lr"] == pytest.approx(1e-4)
    assert cfg["cache"]["budget"] == 0.25 and cfg["flag"] is True
    h1 = config_hash(cfg)
    assert h1 == config_hash(json.loads(json.dumps(cfg)))
    cfg2 = deep_update(cfg, {"rl": {"lr": 3}})
    assert cfg2["rl"]["lr"] == 3 and cfg["rl"]["lr"] != 3
    assert get_dotted(cfg, "rl.gamma") == 0.99 and get_dotted(cfg, "nope.x", 7) == 7
    set_dotted(cfg, "a.b.c", 1)
    assert cfg["a"]["b"]["c"] == 1
    assert parse_override("x.y=[1,2]") == ("x.y", [1, 2])
    with pytest.raises(ValueError):
        parse_override("novalue")


def test_tracker_roundtrip(tmp_runs):
    cfg = {"a": 1, "b": {"c": [1, 2]}}
    with start_run("test", cfg, seed=3, device="cpu", runs_dir=tmp_runs) as run:
        run.log(step=0, loss=1.5, tensor=torch.tensor([1.0, 2.0]))
        run.log(step=1, loss=1.0)
        run.save_json("extra.json", {"k": "v"})
        run.finish({"final": 0.5})
    loaded = load_run(run.dir)
    assert loaded["config"] == cfg
    assert loaded["results"]["final"] == 0.5
    assert loaded["meta"]["seed"] == 3 and loaded["meta"]["status"] == "finished"
    assert loaded["metrics"][0]["tensor"] == [1.0, 2.0]
    assert "commit" in loaded["meta"]
    assert len(list_runs(tmp_runs, kind="test")) == 1
    assert list_runs(tmp_runs, kind="other") == []


def test_tracker_marks_failed(tmp_runs):
    with pytest.raises(RuntimeError):
        with start_run("test", {}, runs_dir=tmp_runs):
            raise RuntimeError("boom")
    r = list_runs(tmp_runs)[0]
    assert r["meta"]["status"] == "failed"
    assert "boom" in r["results"]["error"]


def test_device_helpers_cpu():
    dev = resolve_device("cpu")
    synchronize(dev)
    assert memory_stats(dev).allocated_bytes == 0
    assert pick_dtype(dev) is torch.float32
    assert pick_dtype(dev, "fp16") is torch.float16
    with pytest.raises(ValueError):
        pick_dtype(dev, "int9")
    with PeakTracker(dev) as pk:
        pk.sample()
    assert pk.peak_bytes >= 0


def test_stopwatch_and_samples():
    sw = Stopwatch("cpu")
    with sw("a"):
        sum(range(1000))
    with sw("a"):
        pass
    assert sw.counts["a"] == 2 and sw.as_dict()["a_s"] >= 0
    s = Samples()
    for v in [3.0, 1.0, 2.0, 10.0]:
        s.add(v)
    summ = s.summary()
    assert summ["median"] == 2.5 and summ["min"] == 1.0 and summ["n"] == 4


def test_decode_cost_model_fit():
    from kvrl.bench.cost_model import fit_decode_cost

    curve = [{"cache_len": 1024 * k, "decode_ms_per_tok_median": 10 + 2 * k} for k in range(1, 6)]
    m = fit_decode_cost(curve, device="test")
    assert abs(m.ms_per_token_base - 10) < 1e-6 and abs(m.ms_per_token_per_1k - 2) < 1e-6
    assert m.r2 > 0.999 and abs(m.decode_ms(2048) - 14) < 1e-6
