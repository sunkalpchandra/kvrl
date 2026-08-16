PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup lint test test-all fmt clean demo bench train eval collect frontend

setup:            ## create venv + install deps (reuses system torch if present)
	bash setup.sh

lint:             ## ruff
	$(PY) -m ruff check .

fmt:              ## ruff format + fix imports
	$(PY) -m ruff check --fix . && $(PY) -m ruff format .

test:             ## fast tests (no model)
	$(PY) -m pytest -q -m "not slow"

test-all:         ## incl. slow real-model integration tests
	$(PY) -m pytest -q -m "" 

collect:          ## collect traces with the default config
	$(PY) -m kvrl.collect --config configs/collect.yaml

train:            ## train PPO in the simulator
	$(PY) -m kvrl.train --config configs/train.yaml

eval:             ## evaluate controllers on the task suite
	$(PY) -m kvrl.evaluate --config configs/evaluate.yaml

bench:            ## latency / memory benchmark
	$(PY) -m kvrl.benchmark --config configs/benchmark.yaml

demo:             ## one-click terminal demo (real model)
	$(PY) demo.py

frontend:         ## build dashboard
	cd frontend && npm install && npm run build

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
