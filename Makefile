COMPOSE := docker compose -f deploy/compose.yaml
SRC := packages/platform/src modules/doc_extraction/src
TESTS := packages/platform/tests modules/doc_extraction/tests
MODELS_DIR ?= models
VLLM_PRESET ?= qwen3.5-4b-awq
ALL_PROFILES := --profile gpu --profile observability

.PHONY: install fmt lint typecheck imports test check up up-dev down smoke check-egress models

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

# Full stack: api on real vLLM (preset VLLM_PRESET) + Phoenix tracing.
up:
	VLLM_PRESET=$(VLLM_PRESET) LLMP_LLM_BACKEND=vllm \
	LLMP_OTLP_TRACES_ENDPOINT=http://phoenix:6006/v1/traces \
	$(COMPOSE) $(ALL_PROFILES) up --build --wait

up-dev:
	$(COMPOSE) up --build --wait

down:
	$(COMPOSE) $(ALL_PROFILES) down

# One schema-constrained request through the traced client inside the api container.
smoke:
	$(COMPOSE) exec -T api python -m llmp.smoke

# Must fail to connect: vLLM sits on an internal-only network.
check-egress:
	@if $(COMPOSE) exec -T vllm python3 -c "import socket; socket.create_connection(('1.1.1.1', 443), timeout=5)"; \
	then echo "FAIL: vllm reached the internet"; exit 1; else echo "OK: vllm has no egress"; fi

# One-time, needs network: download pinned model snapshots (see deploy/vllm/models.txt).
models:
	grep -v '^#' deploy/vllm/models.txt | while read -r name repo rev; do \
		uvx --from 'huggingface_hub==2.0.0' hf download "$$repo" --revision "$$rev" \
			--local-dir "$(MODELS_DIR)/$$name" || exit 1; \
	done
