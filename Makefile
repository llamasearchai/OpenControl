# OpenControl Development Makefile
# Author: Nik Jois <nikjois@llamasearch.ai>

.PHONY: help install install-dev test test-all test-unit test-integration test-benchmark
.PHONY: lint format type-check clean build publish docs serve-docs
.PHONY: docker-build docker-run setup-hooks pre-commit run-interactive

# Default target
help:
	@echo "OpenControl Development Commands"
	@echo "==============================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install       - Install package in editable mode"
	@echo "  install-dev   - Install with development dependencies"
	@echo "  setup-hooks   - Install pre-commit hooks"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test          - Run basic test suite"
	@echo "  test-all      - Run all tests with coverage"
	@echo "  test-unit     - Run unit tests only"
	@echo "  test-integration - Run integration tests"
	@echo "  test-benchmark - Run performance benchmarks"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint          - Run all linters (black, isort, flake8, mypy)"
	@echo "  format        - Format code with black and isort"
	@echo "  type-check    - Run mypy type checking"
	@echo "  pre-commit    - Run pre-commit on all files"
	@echo ""
	@echo "Build Commands:"
	@echo "  clean         - Clean build artifacts"
	@echo "  build         - Build package"
	@echo "  publish       - Publish to PyPI (requires credentials)"
	@echo ""
	@echo "Documentation Commands:"
	@echo "  docs          - Build documentation"
	@echo "  serve-docs    - Serve documentation locally"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run Docker container"
	@echo ""
	@echo "Application Commands:"
	@echo "  run-interactive - Start OpenControl interactive mode"

# Installation commands
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs,monitoring,visualization]"

setup-hooks:
	pre-commit install
	pre-commit install --hook-type commit-msg

# Testing commands
test:
	python test_system.py

test-all:
	pytest -v --cov=opencontrol --cov-report=html --cov-report=term-missing --cov-report=xml

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-benchmark:
	pytest -m benchmark --benchmark-only

# Code quality commands
lint: black-check isort-check flake8 mypy

format:
	black opencontrol/ tests/ scripts/
	isort opencontrol/ tests/ scripts/

black-check:
	black --check opencontrol/ tests/ scripts/

isort-check:
	isort --check-only opencontrol/ tests/ scripts/

flake8:
	flake8 opencontrol/ tests/ scripts/

mypy:
	mypy opencontrol/

ruff:
	ruff check opencontrol/ tests/ scripts/

ruff-fix:
	ruff check --fix opencontrol/ tests/ scripts/

pre-commit:
	pre-commit run --all-files

# Build commands
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish: build
	python -m twine upload dist/*

# Documentation commands
docs:
	cd docs && make html

serve-docs:
	cd docs/_build/html && python -m http.server 8080

# Docker commands
docker-build:
	docker build -t opencontrol:latest -f docker/Dockerfile.inference .

docker-run:
	docker run --rm -it -p 8000:8000 opencontrol:latest

docker-dev:
	docker build -t opencontrol:dev -f docker/Dockerfile.dev .
	docker run --rm -it -v $(PWD):/workspace opencontrol:dev

# Application commands
run-interactive:
	opencontrol interactive

run-train:
	opencontrol train --config configs/models/development.yaml

run-evaluate:
	opencontrol evaluate --config configs/models/development.yaml

run-serve:
	opencontrol serve --config configs/models/development.yaml

# Development workflow shortcuts
dev-setup: install-dev setup-hooks
	@echo "Development environment setup complete!"

dev-check: format lint test
	@echo "Development checks passed!"

ci-check: lint test-all
	@echo "CI checks passed!"

# System requirements check
check-requirements:
	@echo "Checking system requirements..."
	@python -c "import sys; print(f'Python version: {sys.version}')"
	@python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	@python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null || echo "CUDA not available"

# Performance profiling
profile:
	python -m cProfile -o profile.prof scripts/profile_model.py
	python -c "import pstats; p = pstats.Stats('profile.prof'); p.sort_stats('cumulative').print_stats(20)"

# Security check
security:
	safety check
	bandit -r opencontrol/

# Update dependencies
update-deps:
	pip-compile requirements.in --upgrade
	pip-compile requirements-dev.in --upgrade

# Database/cache cleanup
clean-cache:
	rm -rf .cache/
	rm -rf logs/
	rm -rf checkpoints/
	rm -rf data/cache/
	rm -rf wandb/

# Full reset (use with caution)
reset: clean clean-cache
	rm -rf .venv/
	@echo "Full reset complete. Run 'make dev-setup' to reinstall."

# Quick development cycle
quick: format lint test
	@echo "Quick development cycle complete!"

# Release preparation
prepare-release: clean dev-check docs
	@echo "Release preparation complete!"
	@echo "Ready to tag and publish!" 