.PHONY: help install format lint test build clean \
        docker-build docker-run docker-test \
        dc-up dc-down dc-logs

# -------------------------------
# User-facing commands
# -------------------------------

help:
	@echo "Available commands:"
	@echo "  make install       - Install project with dev dependencies"
	@echo "  make format        - Run black and isort"
	@echo "  make lint          - Run ruff linting"
	@echo "  make test          - Run pytest"
	@echo "  make build         - Build wheel package"
	@echo "  make clean         - Remove build artifacts and caches"
	@echo ""
	@echo "Docker/devcontainer commands:"
	@echo "  make docker-build  - Build dev Docker image"
	@echo "  make docker-run    - Run a container with mounted workspace"
	@echo "  make docker-test   - Run pytest inside container"
	@echo "  make dc-up         - Start docker-compose services"
	@echo "  make dc-down       - Stop docker-compose services"
	@echo "  make dc-logs       - Tail docker-compose logs"

install:
	poetry install --with dev

format:
	poetry run black .
	poetry run isort .

lint:
	poetry run ruff check .

test:
	poetry run pytest -q

build:
	poetry build

clean:
	rm -rf dist/
	rm -rf build/
	rm -rf .pytest_cache/

# -------------------------------
# Docker helpers
# -------------------------------

docker-build:
	docker build -f docker/Dockerfile.dev -t quant-platform-dev:latest .

docker-run:
	docker run --rm -it -v ${PWD}:/workspace -w /workspace -p 8000:8000 quant-platform-dev:latest bash

docker-test:
	poetry run pytest -q

dc-up:
	docker-compose up -d

dc-down:
	docker-compose down

dc-logs:
	docker-compose logs -f
