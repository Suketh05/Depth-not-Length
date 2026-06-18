#!/usr/bin/env bash
# One-command reproduction of the BriefBench results (Linux/macOS).
# Syncs the environment, builds the native C kernel, runs the test gate, executes the
# benchmark, and regenerates the tables, report, and figures.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

echo "==> syncing environment (uv)"
uv sync --extra dev --extra viz --extra anthropic

echo "==> building native C/NEON kernel"
make -C native/c

echo "==> running the offline test gate"
uv run pytest -m "not live and not neural" -q

echo "==> running the benchmark (synthetic depth crossover)"
uv run membench run --dataset synthetic --budget 150 --out results/rows.jsonl

echo "==> rendering tables, report, figures"
uv run membench tables  --src results/rows.jsonl
uv run membench report  --src results/rows.jsonl --out results/report.md
uv run membench figures --src results/rows.jsonl --out-dir figures

echo "==> done: results/rows.jsonl, results/report.md, figures/"
