# =============================================================================
# Makefile — Survey ML Toolkit Developer Commands
# =============================================================================

.PHONY: help install dev test test-fast test-slow lint format type-check
.PHONY: coverage clean build publish docs sample

PYTHON := python
PIP := pip
PYTEST := pytest

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*
$$
' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n",
$$
1, $\$2}'

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

install: ## Install package
	$(PIP) install -e .

dev: ## Install with all development dependencies
	$(PIP) install -e ".[full]"
	pre-commit install

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test: ## Run all tests
	$(PYTEST) tests/ -v --timeout=300
	@echo "✅ All tests passed"

test-fast: ## Run fast tests only (skip ML training)
	$(PYTEST) tests/ -v -m "not slow" --timeout=60
	@echo "✅ Fast tests passed"

test-slow: ## Run slow tests only (ML training)
	$(PYTEST) tests/ -v -m "slow" --timeout=300
	@echo "✅ Slow tests passed"

coverage: ## Run tests with coverage report
	$(PYTEST) tests/ \
		--cov=survey_toolkit \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-report=xml:coverage.xml \
		-v --timeout=300
	@echo "✅ Coverage report: htmlcov/index.html"

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------

lint: ## Run linter
	ruff check survey_toolkit/ tests/
	@echo "✅ Linting passed"

format: ## Format code with Black
	black survey_toolkit/ tests/ notebooks/
	ruff check --fix survey_toolkit/ tests/
	@echo "✅ Formatting complete"

format-check: ## Check formatting without changes
	black --check --diff survey_toolkit/ tests/
	@echo "✅ Formatting check passed"

type-check: ## Run type checking
	mypy survey_toolkit/ --ignore-missing-imports
	@echo "✅ Type checking complete"

quality: lint format-check type-check ## Run all quality checks
	@echo "✅ All quality checks passed"

# ---------------------------------------------------------------------------
# Build & Release
# ---------------------------------------------------------------------------

build: clean ## Build package
	$(PYTHON) -m build
	twine check dist/*
	@echo "✅ Package built: dist/"

publish-test: build ## Publish to Test PyPI
	twine upload --repository testpypi dist/*
	@echo "✅ Published to Test PyPI"

publish: build ## Publish to PyPI
	twine upload dist/*
	@echo "✅ Published to PyPI"

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

docs: ## Build documentation
	cd docs && make html
	@echo "✅ Docs built: docs/_build/html/"

docs-serve: docs ## Build and serve docs locally
	cd docs/_build/html && $(PYTHON) -m http.server 8000

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

sample: ## Generate sample survey data
	$(PYTHON) -c "\
		from survey_toolkit.utils import generate_sample_survey; \
		df = generate_sample_survey(save_path='data/sample_survey.csv'); \
		print(f'Generated {len(df)} respondents → data/sample_survey.csv')"

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf outputs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleaned"

clean-all: clean ## Clean everything including data
	rm -rf data/sample_survey.csv
	@echo "✅ Deep clean complete"