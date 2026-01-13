.PHONY: all run optimize tensorboard test lint format type-check check clean help install install-dev docker-build docker-run docs

all: run

run:
	@echo "Running RL Order Execution..."
	uv run python main.py

optimize:
	@echo "Running Optuna Hyperparameter Tuning..."
	uv run python src/rl_order_execution/optimize.py

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
	rm -rf runs

docker-build:
	@echo "Building Docker image..."
	docker build -t rl-order-execution .

docker-run:
	@echo "Running Docker container..."
	# Mounts current directory to /app/output to persist plots
	docker run --rm -v "$(shell pwd):/app/output" rl-order-execution

docs:
	@echo "Updating README configuration table..."
	uv run settings-doc generate \
		--class rl_order_execution.settings.Settings \
		--templates config \
		--output-format markdown \
		--between "<!-- settings-start -->" "<!-- settings-end -->" \
		--update README.md \
		--heading-offset 2
	@echo "README.md updated."

help:
	@echo "Available commands:"
	@echo "  make run          - Run the simulation"
	@echo "  make optimize     - Run Optuna hyperparameter tuning"
	@echo "  make tensorboard  - Launch TensorBoard server"
	@echo "  make check        - Run all quality checks (lint + type-check + test)"
	@echo "  make test         - Run unit tests"
	@echo "  make lint         - Check code style"
	@echo "  make type-check   - Run static type checking with mypy"
	@echo "  make format       - Auto-format code"
	@echo "  make docs         - Locally update README config table"
	@echo "  make install      - Install base dependencies"
	@echo "  make install-dev  - Install all dev dependencies"
	@echo "  make docker-build - Build the Docker image"
	@echo "  make docker-run   - Run the Docker container"
	@echo "  make clean        - Remove virtualenv, caches, and plots"