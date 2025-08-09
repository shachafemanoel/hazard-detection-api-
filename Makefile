# Hazard Detection API - Development Makefile

.PHONY: help install dev-install lint format typecheck test run docker-build docker-run clean

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  dev-install  - Install development dependencies and pre-commit hooks"
	@echo "  lint         - Run ruff linter"
	@echo "  format       - Format code with black and ruff"
	@echo "  typecheck    - Run mypy type checker"
	@echo "  test         - Run pytest tests"
	@echo "  run          - Start development server"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  clean        - Clean up cache files"

# Installation
install:
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt

dev-install: install
	pip install ruff black mypy pre-commit pytest-cov
	pre-commit install
	@echo "✅ Development environment ready!"

# Code quality
lint:
	ruff check app/ main.py

format:
	black app/ main.py
	ruff format app/ main.py

typecheck:
	mypy app/ main.py

# Testing  
test:
	pytest -v

test-cov:
	pytest --cov=app --cov-report=html --cov-report=term

# Development server
run:
	python main.py

# Docker commands
docker-build:
	docker build -t hazard-detection-api .

docker-run:
	docker run -p 8080:8080 --env-file .env hazard-detection-api

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

# All checks (for CI)
check-all: lint typecheck test
	@echo "✅ All checks passed!"