# ============================================================
# Claim2Code — entrypoint
# Usage: `make help`
# All targets work from project root. Variables overridable on CLI.
# ============================================================

PROJECT_ROOT ?= $(shell pwd)
COMPOSE      ?= docker compose
HOSTNAME_S   ?= $(shell hostname -s 2>/dev/null || echo unknown)
DATE_UTC     ?= $(shell date -u +%Y-%m-%dT%H-%M-%SZ)

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message.
	@echo "Claim2Code — make targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Override variables:"
	@echo "  PROJECT_ROOT=/path  COMPOSE='docker compose -f file.yml'"
	@echo ""

# ---------------------------- Lifecycle ----------------------------

.PHONY: dev
dev: ## Start all services (api + web + postgres+pgvector).
	@bash scripts/dev-up.sh

.PHONY: down
down: ## Stop all services (volumes preserved).
	@$(COMPOSE) down

.PHONY: reset
reset: ## Stop + remove volumes + clear .volumes/  (DESTRUCTIVE: data only).
	@echo "This will DELETE all database/storage data (code is untouched)."
	@read -p "Type 'yes' to continue: " ans && [ "$$ans" = "yes" ]
	@$(COMPOSE) down -v
	@rm -rf .volumes/

# ---------------------------- Build / Test ----------------------------

.PHONY: build
build: ## Build API + Frontend images.
	@$(COMPOSE) build

.PHONY: test
test: ## Run all unit tests (Python + frontend).
	@cd apps/api && python -m pytest
	@cd apps/web && npm test --silent

.PHONY: test-integration
test-integration: ## Run integration tests (testcontainers).
	@cd apps/api && python -m pytest -m integration

.PHONY: lint
lint: ## Lint Python + frontend.
	@cd apps/api && ruff check . && mypy .
	@cd apps/web && npm run lint

.PHONY: format
format: ## Format Python code with ruff.
	@cd apps/api && ruff format .

# ---------------------------- Migrations ----------------------------

.PHONY: migrate
migrate: ## Apply database migrations.
	@cd apps/api && alembic upgrade head

.PHONY: migration-new
migration-new: ## Create a new migration (usage: make migration-new NAME=...).
	@cd apps/api && alembic revision --autogenerate -m "$(NAME)"

# ---------------------------- Eval ----------------------------

# Each evaluator writes evaluation/results/<name>.json; `make eval-report`
# folds them into evaluation/results.json + evaluation/report.md.

.PHONY: eval-retrieval
eval-retrieval: ## Run retrieval evaluation (Recall@1/5, MRR).
	@python evaluation/evaluate_retrieval.py

.PHONY: eval-verification
eval-verification: ## Run verification evaluation (accuracy, discrepancy P/R, grounding). Slow: one LLM call per claim; ARGS=--resume continues.
	@python evaluation/evaluate_verification.py $(ARGS)

.PHONY: eval-safety
eval-safety: ## Run the safety evaluator (secret / injection corpus through SafetyGate).
	@python evaluation/evaluate_safety.py

.PHONY: eval-report
eval-report: ## Render evaluation/results.json + report.md from the last runs.
	@python evaluation/report.py

.PHONY: eval-all
eval-all: ## Run every evaluator + render the report.
	@python evaluation/evaluate_all.py

.PHONY: eval-local
eval-local: ## Preflight (DB, indexed pairs, served models), then every evaluator + report; ARGS=--resume | --check.
	@bash scripts/eval-local.sh $(ARGS)

# ---------------------------- Logs ----------------------------

.PHONY: logs
logs: ## Tail logs from all services.
	@$(COMPOSE) logs -f

.PHONY: logs-api
logs-api: ## Tail API logs only.
	@$(COMPOSE) logs -f api

# ---------------------------- Bootstrap ----------------------------

.PHONY: secrets
secrets: ## Generate strong random values for .env secrets.
	@echo "Suggested secrets (copy into .env):"
	@echo "  POSTGRES_PASSWORD=$$(openssl rand -hex 24)"
	@echo "  OPENAI_API_KEY=sk-keep-existing-if-you-have-one"