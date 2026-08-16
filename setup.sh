#!/usr/bin/env bash
# Reproducible environment setup for kvrl.
#   ./setup.sh            -> creates .venv (reusing system torch if importable) and installs deps
#   KVRL_CLEAN=1 ./setup.sh -> fully isolated venv (downloads torch)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
if [ "${KVRL_CLEAN:-0}" = "1" ]; then
  echo "[setup] creating isolated venv"
  $PYTHON -m venv .venv
else
  if $PYTHON -c "import torch" 2>/dev/null; then
    echo "[setup] system torch found -> venv with --system-site-packages (saves disk)"
    $PYTHON -m venv --system-site-packages .venv
  else
    echo "[setup] no system torch -> isolated venv"
    $PYTHON -m venv .venv
  fi
fi
.venv/bin/pip install --upgrade pip >/dev/null
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . >/dev/null
.venv/bin/python - <<'PY'
import torch, transformers, platform
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[setup] python {platform.python_version()} torch {torch.__version__} transformers {transformers.__version__} device={dev}")
PY
echo "[setup] done. Next: make test ; python demo.py"
