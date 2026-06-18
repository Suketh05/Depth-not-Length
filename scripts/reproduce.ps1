#Requires -Version 7.0
# One-command reproduction of the BriefBench results (Windows / PowerShell).
# The native C kernel is platform-specific and optional (the Python pipeline falls
# back to NumPy), so it is skipped here; everything else mirrors reproduce.sh.
$ErrorActionPreference = "Stop"

Write-Host "==> syncing environment (uv)"
uv sync --extra dev --extra viz --extra anthropic

Write-Host "==> running the offline test gate"
uv run pytest -m "not live and not neural" -q

Write-Host "==> running the benchmark (synthetic depth crossover)"
uv run membench run --dataset synthetic --budget 150 --out results/rows.jsonl

Write-Host "==> rendering tables, report, figures"
uv run membench tables  --src results/rows.jsonl
uv run membench report  --src results/rows.jsonl --out results/report.md
uv run membench figures --src results/rows.jsonl --out-dir figures

Write-Host "==> done: results/rows.jsonl, results/report.md, figures/"
