# Reproducible environment for BriefBench. Builds the Python package and the native
# C kernel, then exposes the `membench` CLI. The reported numbers come from the
# deterministic CPU/NumPy path, so this image reproduces them without a GPU.
#
#   docker build -t briefbench .
#   docker run --rm briefbench run --dataset synthetic
FROM python:3.12-slim

# Pinned (DL3008) for a reproducible build; versions track the python:3.12-slim base
# (Debian trixie). Bump in lockstep when the base image moves.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential=12.12 \
      make=4.4.1-2 \
      git=1:2.47.3-0+deb13u1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Single RUN (DL3059) with a pinned uv (DL3013): install uv, sync the frozen env, and
# build the native C kernel so the deterministic parity path is available at runtime.
RUN pip install --no-cache-dir uv==0.11.21 \
 && uv sync --extra dev --extra viz --extra anthropic --frozen \
 && make -C native/c

ENTRYPOINT ["uv", "run", "membench"]
CMD ["run", "--dataset", "synthetic"]
