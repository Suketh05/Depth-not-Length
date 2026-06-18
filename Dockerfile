# Reproducible environment for BriefBench. Builds the Python package and the native
# C kernel, then exposes the `membench` CLI. The reported numbers come from the
# deterministic CPU/NumPy path, so this image reproduces them without a GPU.
#
#   docker build -t briefbench .
#   docker run --rm briefbench run --dataset synthetic
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential make git \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

WORKDIR /app
COPY . .

RUN uv sync --extra dev --extra viz --extra anthropic --frozen
RUN make -C native/c

ENTRYPOINT ["uv", "run", "membench"]
CMD ["run", "--dataset", "synthetic"]
