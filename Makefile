.PHONY: install validate validate-all validate-fix validate-incremental validate-report \
        validate-parallel validate-parallel-fix \
        sync sync-dry-run sync-incremental test test-cov format lint typecheck check help

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# Installation
# =============================================================================

install:  ## Install dependencies with uv
	uv sync

install-dev:  ## Install with dev dependencies
	uv sync --dev

# =============================================================================
# Validation targets
# =============================================================================

validate: validate-all  ## Validate all notes (alias for validate-all)

validate-all:  ## Validate all notes in vault
	uv run obsidian-anki-sync validate all

validate-fix:  ## Validate and auto-fix safe issues
	uv run obsidian-anki-sync validate all --fix

validate-incremental:  ## Only validate changed files
	uv run obsidian-anki-sync validate all --incremental

validate-report:  ## Generate markdown validation report
	uv run obsidian-anki-sync validate all --report validation-report.md

validate-verbose:  ## Validate with per-file details
	uv run obsidian-anki-sync validate all --verbose

validate-fix-verbose:  ## Validate, fix, and show details
	uv run obsidian-anki-sync validate all --fix --verbose

validate-stats:  ## Show validation cache statistics
	uv run obsidian-anki-sync validate stats

validate-clear-cache:  ## Clear validation cache
	uv run obsidian-anki-sync validate clear-cache

validate-parallel:  ## Validate all notes using parallel processing
	uv run obsidian-anki-sync validate all --parallel

validate-parallel-fix:  ## Validate and auto-fix using parallel processing
	uv run obsidian-anki-sync validate all --parallel --fix

# =============================================================================
# Sync targets
# =============================================================================

sync:  ## Sync notes to Anki
	uv run obsidian-anki-sync sync

sync-dry-run:  ## Preview sync without applying changes
	uv run obsidian-anki-sync sync --dry-run

sync-incremental:  ## Sync only new notes
	uv run obsidian-anki-sync sync --incremental

sync-sample:  ## Sync sample of 10 notes (dry-run)
	uv run obsidian-anki-sync test-run --count 10

# =============================================================================
# Development targets
# =============================================================================

test:  ## Run tests
	uv run pytest

test-cov:  ## Run tests with coverage report
	uv run pytest --cov

test-fast:  ## Run tests excluding slow ones
	uv run pytest -m "not slow"

format:  ## Format code with ruff and isort
	uv run ruff format . && uv run isort .

lint:  ## Run ruff linter
	uv run ruff check .

lint-fix:  ## Run ruff linter with auto-fix
	uv run ruff check . --fix

typecheck:  ## Run mypy type checker
	uv run mypy src/

check: format lint typecheck test  ## Run all quality checks (format, lint, typecheck, test)

# =============================================================================
# Utility targets
# =============================================================================

check-setup:  ## Run pre-flight checks
	uv run obsidian-anki-sync check

decks:  ## List available Anki decks
	uv run obsidian-anki-sync decks

models:  ## List available Anki note types
	uv run obsidian-anki-sync models

logs:  ## Analyze recent logs
	uv run obsidian-anki-sync analyze-logs

progress:  ## Show sync progress
	uv run obsidian-anki-sync progress

index:  ## Show vault index statistics
	uv run obsidian-anki-sync index

# =============================================================================
# Help
# =============================================================================

help:  ## Show this help message
	@echo "Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make install         # Install dependencies"
	@echo "  make validate        # Validate all notes"
	@echo "  make validate-fix    # Validate and auto-fix"
	@echo "  make sync-dry-run    # Preview sync changes"
	@echo "  make check           # Run all quality checks"
