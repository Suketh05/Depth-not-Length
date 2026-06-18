# Developer entrypoints. All commands run inside the uv-managed environment.
.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Sync the dev environment (core + dev + viz + anthropic)
	$(UV) sync --extra dev --extra viz --extra anthropic

.PHONY: lint
lint: ## Ruff lint + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

.PHONY: format
format: ## Auto-format with ruff
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

.PHONY: type
type: ## Strict mypy over the package
	$(UV) run mypy src

.PHONY: test
test: ## Run the offline test suite (excludes live/neural)
	$(UV) run pytest -m "not live and not neural"

.PHONY: test-all
test-all: ## Run every test including neural (needs the neural extra)
	$(UV) run pytest

.PHONY: cov
cov: ## Coverage report
	$(UV) run pytest -m "not live and not neural" --cov --cov-report=term-missing

.PHONY: check
check: lint type test ## Full pre-commit gate (lint + type + test)

.PHONY: run
run: ## Run the full benchmark with the default (offline) config
	$(UV) run membench run

.PHONY: tables
tables: ## Render result tables from results/
	$(UV) run membench tables

.PHONY: report
report: ## Generate the full analysis report
	$(UV) run membench report

.PHONY: clean
clean: ## Remove caches and generated artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov results figures
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
