COMPOSE := docker compose -f deploy/compose.yaml
SRC := packages/platform/src modules/doc_extraction/src
TESTS := packages/platform/tests modules/doc_extraction/tests

.PHONY: install fmt lint typecheck imports test check up-dev down

install:
	uv sync --all-packages

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy $(SRC) $(TESTS)

imports:
	uv run lint-imports

test:
	uv run pytest

check: lint typecheck imports test

up-dev:
	$(COMPOSE) up --build --wait

down:
	$(COMPOSE) down
