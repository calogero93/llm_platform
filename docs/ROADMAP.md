# Roadmap

Budget: 5–10 h/week. Each phase is sized for 1–3 weeks and ends with a review stop:
tests + evals run, summary of what was built, numbers, open issues. No phase starts before the
previous one is approved.

Acceptance criteria are written to be checked by a command. Quality thresholds marked
*provisional* are revised after the first real measurement — they exist so that "done" is not
open-ended, not because the numbers are known in advance.

---

## Phase 0 — Skeleton, tooling, CI, LLM client with mock

Scope: uv workspace (`packages/platform`, `modules/doc_extraction` stub), settings, module
registry via entry points, `LLMClient` protocol + `VLLMClient` + `MockLLMClient` (`schema` and
`replay` modes), prompt store, JSON logging with correlation id, FastAPI app factory with
`/health`, Makefile, `deploy/compose.yaml` with the `dev` profile, GitHub Actions.

Acceptance:
- [x] `make test` green; `make lint` (ruff check + ruff format --check) and `make typecheck`
      (mypy --strict) clean.
- [x] `lint-imports` passes: `llmp` imports nothing from `modules/`.
- [x] A test module registered only via entry point is discovered and its router mounted,
      with zero changes to `packages/platform`.
- [x] `MockLLMClient(replay)` returns identical output for identical requests and raises a
      clear error on cassette miss; `MockLLMClient(schema)` returns an instance that validates
      against the requested pydantic model.
- [x] `VLLMClient` tested against a fake HTTP server (`httpx.MockTransport`): request shape
      incl. `response_format.json_schema`, usage parsing, timeout and 5xx handling.
- [x] `docker compose --profile dev up` → `curl /health` returns 200; every log line is valid
      JSON containing `correlation_id`; `X-Request-ID` is echoed.
- [x] GitHub Actions runs lint, typecheck, tests, import contract, gitleaks on push/PR — green.

## Phase 1 — vLLM on the local GPU, first traced call end to end

Scope: `make models` (one-time download to a volume), `gpu` profile, sm_120 smoke test, memory
tuning of candidates A and B (see ARCHITECTURE §2.2), Phoenix in `observability` profile, OTel
tracing decorator on `LLMClient`, `/health` reporting upstream readiness.

Acceptance:
- [x] `vllm/vllm-openai:v0.30.0` starts on the RTX 5070 with model A and with model B; startup
      log numbers (weights GiB, KV blocks, max concurrency at `max_model_len`) recorded in
      `docs/benchmarks/serving.md`, replacing the estimates in ARCHITECTURE §2.2.
- [x] A request with `response_format: json_schema` returns output valid against the schema
      (proves structured outputs work on sm_120 with the chosen quantization).
- [x] `make up` then `make smoke`: an API call hits vLLM and produces a Phoenix trace with model,
      prompt id/version/hash, input/output tokens, latency; the API log line carries the same
      `trace_id`.
- [x] `/health` returns 503 while vLLM is loading and 200 once ready.
- [x] Egress check: `docker compose exec vllm python -c "<connect to 1.1.1.1:443>"` fails;
      `HF_HUB_OFFLINE=1` set and startup succeeds without network.
- [x] Phase review includes a go/no-go on the primary model. (2026-09-27: model A confirmed.)

## Phase 2 — Synthetic data generator + evaluation harness (+ deterministic XML parser)

Scope: generator for order → DDT → invoice chains (XML, native PDF, degraded scan) with
ground truth; platform eval harness (runner, metrics, reports, compare, baseline gating).
Changes agreed at phase start: the deterministic FatturaPA parser (formerly 3a) moved here so
`make eval` has a real predictor, whose F1 must be exactly 1.0 (end-to-end check of generator,
parser and metrics); cassette recording moved to 3b, where the first LLM suite needs it.

Acceptance:
- [x] `make synth` is deterministic: regenerated files match the SHA-256 in `manifest.json`
      (`make synth-check`; CI regenerates and runs `git diff --exit-code`).
- [x] 100% of generated FatturaPA XML files validate against the vendored XSD FPR12 1.2.3.
- [x] Generated P.IVA numbers pass the check digit; each invoice satisfies Σ lines = imponibile
      and imposta = imponibile × aliquota per VAT bucket (half-up rounding to cents) — tested.
- [x] Degradation levels (`clean`, `light`, `heavy`) are parameters; contact sheet committed
      (`evals/datasets/chains/v1/contact_sheet.jpg`).
- [x] Datasets `ci` (30 chains) and `full` (300 chains) versioned as `chains/v1`; ground truth and
      manifests committed, documents regenerated.
- [x] Harness unit-tested on toy predictions with hand-computed P/R/F1.
- [x] `make eval` writes `report.json` with the metadata of ARCHITECTURE §2.7;
      `make eval-compare A=... B=...` prints a markdown diff table.
- [x] CI runs the eval and gates it against the committed baseline; a test demonstrates that a
      deliberate regression fails the gate.

## Phase 3 — Extraction: XML → native PDF → scans, measured at each step

### 3a — FatturaPA XML (deterministic)
- [x] Parser done in Phase 2 (F1 = 1.00 on `ci` and `full`, multi-body files, DTD rejected).
- [ ] `POST /doc-extraction/extract` accepts XML and returns the extracted `Document`.

### 3b — Native PDF (Docling + LLM)
- [ ] Cassette recording (`make eval-record`) so CI replays real vLLM outputs.
- [ ] Prompt `extract_invoice/v1` + schema-constrained generation; results for models A and B.
- [ ] Eval report per field on `full` native-PDF split, committed as baseline.
- [ ] *Provisional* target: macro field F1 ≥ 0.95 (header fields), ≥ 0.90 (line items).
- [ ] Confidence stage 1 (validators + grounding) implemented; ECE reported.
- [ ] p50/p95 latency and €/document reported, split by stage (parse vs LLM).

### 3c — Scanned PDF (OCR)
- [ ] OCR engine chosen by measurement (Docling default vs Tesseract `ita`), both numbers reported.
- [ ] F1 reported per degradation level; *provisional* target ≥ 0.85 on `light`.
- [ ] Confidence stage 2 (OCR confidence) added if stage 1 ECE > 0.10 on scans; stage 3
      (logprobs) only if still > 0.10.
- [ ] DDT and purchase orders extracted with the same pipeline (own schemas + prompts).

## Phase 4 — Reconciliation, guardrails, prompt-injection suite

- [ ] `POST /doc-extraction/reconcile` takes a set of documents (or their extractions) and returns
      links + typed discrepancies.
- [ ] Discrepancy detection P/R/F1 per type on the synthetic chains (ground truth = injected
      discrepancies); *provisional* target F1 ≥ 0.95 on XML/native inputs.
- [ ] Decision recorded (with numbers) on whether an LLM is needed for line matching.
- [ ] Injection suite ≥ 6 attack types (ARCHITECTURE §2.8), ≥ 10 docs each; report attack success
      rate (ASR) per type = share of docs where an attacker-chosen value reaches the output
      **without** being flagged. Target: flagged-or-blocked for 100% of arithmetic-breaking attacks.
- [ ] Hidden-text detector for native PDFs with its own P/R.
- [ ] Injection suite runs in CI (replay) and locally (real vLLM).

## Phase 5 — Demo UI

A small web page to show the extraction module end to end without curl: the thing a reviewer or
a prospective customer actually looks at. Demo-grade, not a product UI: no auth, bound to
`127.0.0.1` like the rest of the dev stack.

Approach (confirm at phase start): server-rendered pages from the `doc_extraction` module
(FastAPI + Jinja2 templates + htmx, vendored as a static file). No Node build, no separate
container, no CDN — it must work on an offline customer server. Alternative considered: a
React/TypeScript SPA (showcases frontend skills, but adds a build toolchain and a second
container for a portfolio whose focus is ML systems).

Acceptance:
- [ ] `GET /doc-extraction/ui`: upload an order, a DDT and an invoice (XML, PDF or scan), or
      pick a ready-made synthetic set from the `ci` dataset.
- [ ] Extracted fields shown per document with their confidence; low-confidence fields and
      failed validators (P.IVA, arithmetic) highlighted.
- [ ] Reconciliation shown as a table of typed discrepancies linked to the lines involved.
- [ ] Each document links to its Phoenix trace (trace id from the response).
- [ ] Works fully offline: a test asserts rendered pages reference no external URLs; all
      static assets are served by the api.
- [ ] Upload limits enforced (size, count, content type); tested.
- [ ] Endpoint tests with the mock LLM; a screenshot/GIF committed for the README.

## Phase 6 — Observability dashboard, load test, benchmark report

- [ ] Grafana dashboard provisioned from repo: tokens/s, TTFT p50/p95, e2e latency p50/p95,
      KV-cache usage, running/waiting requests, API documents/s, stage latency. Metric names
      verified against vLLM 0.30.0 `/metrics`.
- [ ] `make loadtest`: `vllm bench serve` sweep over concurrency (1, 2, 4, 8) with realistic
      prompt/output lengths from the eval set; locust scenario for end-to-end extraction.
- [ ] `gpu_hour_cost_eur` → €/1M tokens derived from measured throughput, written to config.
- [ ] README "Benchmarks" section: model A vs B quality (per-field F1), latency, throughput,
      €/document, hardware and exact versions — all reproducible by one documented command.

## Phase 7 — Retrieval + RAG module (outline, detailed after Phase 6)

pgvector + TEI CPU services under a `rag` profile; `rag_circolari` module with citation-grounded
answers, retrieval eval (recall@k, MRR), answer faithfulness eval; first agent loop with the
per-module tool allowlist. Validates the "new module without touching the platform" claim:
acceptance = diff of `packages/platform` for this phase is limited to the retrieval client, and
zero changes are needed to register the module.
