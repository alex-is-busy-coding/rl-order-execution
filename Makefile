.PHONY: all run clean help

all: run

run:
	@echo "Running RL Order Execution..."
	uv run python main.py

tensorboard:
	@echo "Launching TensorBoard (http://localhost:6006)..."
	uv run tensorboard --logdir=runs

test:
	@echo "Running tests..."
	uv run pytest tests/

lint:
	@echo "Checking code style..."
	uv run ruff check .
	uv run ruff format --check .

type-check:
	@echo "Running static type checks..."
	uv run mypy src/ tests/

format:
	@echo "Formatting code..."
	uv run ruff format .
	uv run ruff check --fix .

check: lint type-check test

install:
	@echo "Syncing dependencies..."
	uv sync

install-dev:
	@echo "Syncing dev dependencies..."
	uv sync --all-extras --dev

clean:
	@echo "Cleaning up..."
	rm -rf .venv
	rm -rf __pycache__
	rm -f execution_analysis.png

help:
	@echo "Available commands:"
	@echo "  make run          - Run the simulation"
	@echo "  make tensorboard  - Launch TensorBoard server"
	@echo "  make check        - Run all quality checks (lint + type-check + test)"
	@echo "  make test         - Run unit tests"
	@echo "  make lint         - Check code style"
	@echo "  make type-check   - Run static type checking with mypy"
	@echo "  make format       - Auto-format code"
	@echo "  make install      - Install base dependencies"
	@echo "  make install-dev  - Install all dev dependencies"
	@echo "  make clean        - Remove virtualenv, caches, and plots"