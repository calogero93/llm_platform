# CLAUDE.md

On-premise LLM platform for Italian SMEs + pluggable use-case modules. Read
`docs/ARCHITECTURE.md` (decisions) and `docs/ROADMAP.md` (current phase, acceptance criteria)
before changing anything non-trivial.

## Hard rules

- **No data leaves the machine at runtime.** No calls to external APIs/services from runtime
  code or containers. Network access is allowed only at install/build time (`make models`,
  image builds). Keep telemetry env vars disabled in `deploy/compose.yaml`.
- **Platform never imports modules.** `packages/platform` (`llmp`) must not import from
  `modules/*`; enforced by `lint-imports`. Modules get services via `PlatformContext`.
  A new module must be added to `forbidden_modules` in the root `pyproject.toml` contract.
- **No LangChain / LlamaIndex** or similar orchestration frameworks. Orchestration is
  hand-written and thin.
- **Every new runtime dependency** gets a row in the dependency ledger in
  `docs/ARCHITECTURE.md` §4 with a reason. If a stdlib solution is < ~50 lines, use stdlib.
- **No secrets in the repo.** Config via env / `.env` (gitignored); update `.env.example`.
- **No real customer data** anywhere. All test/eval data comes from the synthetic generator.
- **Datasets are immutable per version.** If a generator change alters any output byte, create a
  new dataset version directory instead of overwriting (CI enforces it via manifest hashes).
- **Every LLM call goes through `LLMClient`** (so it is traced) and uses a prompt from the
  prompt store (so it is versioned). No inline prompt strings in pipeline code.
- **Prompts are immutable once used in a committed baseline.** Change = new `vN+1.md` file.
- **Phases end with a stop.** At the end of a roadmap phase: run `make check` and `make eval`,
  summarize work + numbers + open issues, and wait for review.
- Language: code, comments, docs, commit messages in **English**.

## Commands

| Command | What |
|---|---|
| `make install` | `uv sync --all-packages` |
| `make fmt` | ruff fix + format |
| `make lint` | ruff check + format check |
| `make typecheck` | mypy --strict on sources and tests |
| `make imports` | import-linter contracts (platform must not import modules) |
| `make test` | pytest (mock LLM; no GPU, no network) |
| `make check` | lint + typecheck + imports + test (what CI runs) |
| `make models` | one-time download of pinned model snapshots into `./models` (needs network) |
| `make up` | api + vLLM (`VLLM_PRESET`, default `qwen3.5-4b-awq`) + Phoenix; waits until healthy |
| `make up-dev` | api only with mock LLM, on `127.0.0.1:${LLMP_API_PORT:-8080}` |
| `make down` | stop everything (all profiles) |
| `make smoke` | one traced, schema-constrained request through the api container; prints trace id |
| `make check-egress` | asserts the vLLM container cannot reach the internet |

Phoenix UI: `http://127.0.0.1:${PHOENIX_PORT:-6006}`. vLLM presets: `deploy/vllm/presets/*.yaml`.

| `make synth` | regenerate synthetic datasets (`SYNTH_SPLITS=ci full`) + contact sheet |
| `make synth-check` | regenerate `ci` in memory and compare with manifest hashes |
| `make eval` | run an eval suite (`SUITE`, `SPLIT`; mock LLM unless `LLMP_LLM_BACKEND=vllm`) |
| `make eval-gate` | run the suite and fail if a gated metric regressed vs the committed baseline |
| `make eval-baseline` | promote a fresh run to `modules/<m>/evals/baselines/<suite>.json` |
| `make eval-compare A=… B=…` | markdown diff of two eval reports |

Targets planned in `docs/ROADMAP.md` (`eval-record`, `loadtest`) are added in the phase that
introduces them; keep this table in sync.

## Conventions

- Python 3.12, uv workspace. Import names: `llmp` (platform), `doc_extraction` (module).
- Types everywhere; mypy strict must pass. All structured data types are Pydantic v2 models
  (frozen where immutable) — no dataclasses, NamedTuples or TypedDicts.
- Async for I/O paths (HTTP, LLM). CPU-heavy parsing (Docling) runs in a thread/process pool,
  never blocking the event loop.
- Money is `Decimal`, never float. Dates are `datetime.date`. P.IVA stored without `IT` prefix.
- Errors: raise specific exceptions; do not catch-and-ignore. Do not add handling for states
  that cannot happen.
- Logging: `logging.getLogger(__name__)`, structured fields via `extra={...}`; never log full
  document contents at INFO level.
- Tests: pytest, `pytest-asyncio`. Bug fix = failing test first. Unit tests never need GPU or
  network; tests needing vLLM are marked `@pytest.mark.gpu` and skipped in CI.
- Eval metric changes require re-running baselines and noting the change in the phase summary.
- Keep diffs surgical: no drive-by refactors or reformatting of unrelated code.
- Commits: conventional style (`feat:`, `fix:`, `chore:`, `docs:`, `test:`), small and focused.

## Environment notes

- Dev: WSL2 on Windows, RTX 5070 Laptop 8 GB (sm_120), driver 596.21 (CUDA ≤ 13.2), Docker
  engine with NVIDIA runtime. WSL2 memory limit matters: check `free -g` before running the full
  stack.
- vLLM image pinned to `vllm/vllm-openai:v0.30.0`. Do not bump without re-running Phase 1
  smoke tests and the eval baselines.
