# CPU image: runs the fast tests, the simulator, training, and (slowly) the real model on CPU.
# GPU users: swap the torch index for a CUDA wheel; the device abstraction picks CUDA up.
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e . && python -m pytest -q -m "not slow"
ENV KVRL_DEVICE=cpu
CMD ["python", "demo.py", "--tokens", "1024", "--max-new-tokens", "12"]
