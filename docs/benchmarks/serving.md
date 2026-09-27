# Serving benchmarks — Phase 1 (memory fit and first latency numbers)

Measured 2026-09-27. These are fit/sanity numbers, not the Phase 6 load test.

## Environment

| | |
|---|---|
| GPU | RTX 5070 Laptop, 8151 MiB, compute capability 12.0 (sm_120) |
| Driver | 596.21 (Windows host, WSL2), CUDA ≤ 13.2 |
| vLLM | `vllm/vllm-openai:v0.30.0@sha256:8a69ffad…4b90` (CUDA 13.0) |
| Host | WSL2 with 24 GB RAM limit, Docker Engine + NVIDIA runtime |

**WSL2 overhead:** only **6.83 of 7.96 GiB** are free when vLLM starts (WDDM reservation +
CUDA context), so `gpu-memory-utilization` must stay ≤ ~0.85. The initial 0.90/0.92 presets failed
at startup. A native Linux server should leave more headroom; re-measure there.

## Memory: estimates vs measured

| | A: Qwen3.5-4B AWQ (`qwen3.5-4b-awq`) | B: Qwen3-8B AWQ (`qwen3-8b-awq`) |
|---|---|---|
| Estimated weights (on disk) | 4.0 GB | 6.1 GB |
| Measured weights (+ non-torch for A) | 3.4 GiB (vision tower skipped: `language-model-only`) | 5.71 GiB |
| Peak activation | 1.26 GiB (2048-token prefill chunks) | profiler: no room left for KV |
| KV cache | 2.1 GiB (profiled) | 0.5 GiB (fixed `kv-cache-memory`) |
| KV capacity | **56,050 tokens** (est. ~70k) | **7,280 tokens** fp8 (est. 9–12k) |
| Max concurrency at `max-model-len` | **3.42×** at 16,384 | **1.18×** at 6,144 |
| GPU memory in use when serving | ~7.07 GB | ~7.46 GB |
| Measured KV cost | ~40 KiB/token incl. amortized Gated-DeltaNet state (est. 32) | fp8, 72 KiB/token (as estimated) |

## First latency numbers (single request, warm server)

| Request | A | B |
|---|---|---|
| Smoke: schema-constrained JSON, ~100 prompt / 12–15 output tokens | 0.36 s | 0.34 s (0.93 s first call) |
| Long prompt, same text (A: 4,641 tokens, B: 5,761 tokens¹), short answer, cold prefix | **2.1 s** | **4.6 s** |
| Same long prompt again (prefix cache hit) | 0.5 s | — |

¹ Same Italian text; Qwen3.5's 248k-token vocabulary encodes it in ~20% fewer tokens.

Startup: cold start (torch.compile + CUDA graph capture) ~2–3 min for A; restart with the compile
cache volume ~40 s. B spends ~110 s loading weights (AWQ → Marlin repacking).

## Verified on sm_120

- AWQ W4A16 checkpoints (compressed-tensors for A, AutoAWQ format for B) load and run.
- Qwen3.5 Gated-DeltaNet Triton kernels, FlashInfer attention, CUDA graphs, fp8 KV cache.
- Structured outputs (`response_format: json_schema`, xgrammar backend): valid, correct output.

## Decision (proposed): model A is primary

A leaves ~4× the KV capacity of B at 2.7× the context length and is ~2× faster on long prompts. B
runs only with a hand-fixed KV budget and effectively one request at a time. B stays as the
quality baseline for the Phase 3 A/B eval; if B clearly wins on extraction quality, the
trade-off is re-evaluated on hardware with more VRAM.
