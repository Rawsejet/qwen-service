# Qwen3.8-27B Inference Benchmark — vLLM 0.24.0

**Date:** 2026-08-25
**Hardware:** 2x RTX PRO 6000 Blackwell Workstation (SM120, ~96GB each), PCIe (no NVLink)
**Software:** vLLM 0.24.0, conda env `vllm`, CUDA driver 595.58.03
**Model:** Qwen3.8-27B (dense 27B, BF16 ~54GB, `qwen3_5` hybrid GDN arch, vision, MTP)
**Serve flags:** FLASH_ATTN, prefix caching, chunked prefill, `--max-num-seqs 256`, Tools mode, no MTP

Workload: `vllm bench serve`, random dataset, 1024-in / 256-out, seed 42, `--ignore-eos`.

## Baseline — TP=1 (single GPU 0, 0.85 util, 65K ctx)

| Metric | Concurrency 1 (6 req) | Concurrency 8 (48 req) |
|---|---|---|
| Output throughput | 28.1 tok/s | 198.6 tok/s (aggregate) |
| Decode TPOT (median) | 34.98 ms | 35.82 ms |
| Median TTFT | 187 ms | 1250 ms |
| Total token throughput | 140 tok/s | 993 tok/s |

**Key observation:** TPOT is flat 35.0 → 35.8 ms as concurrency goes 1 → 8 while aggregate
throughput scales ~7×. Decode is **memory-bandwidth bound** (reading 54GB BF16 weights per
token step); compute is idle. This is the lever list below.

## Reference point
- Qwen3.6-27B-**FP8** TP=1 (prior benchmark, 2026-07-09): 50.5 tok/s single stream.
  Qwen3.8-27B **BF16** TP=1: 28.1 tok/s ≈ 55% — consistent with FP8 being half the weight bytes.

## Speed levers to test (ranked by expected single-stream gain)
1. **FP8 weights** — decode is BW-bound, so halving weight bytes ≈ up to 2× decode.
   - a) `--quantization fp8` (dynamic, on-the-fly from BF16 at load; zero prep)
   - b) offline llm-compressor W8A8-FP8 checkpoint (best quality/perf, one-time quantize)
2. **MTP speculative decoding** — model ships MTP; `--speculative-config mtp` (script mode 2).
   Prior note: +50–70% on Qwen3.5 single-stream.
3. **TP=2 (dual GPU)** — prior 27B dense: +53% decode by splitting weight reads across 2x BW.
4. **Combined** FP8 + MTP + TP=2 (levers stack; FP8 also frees VRAM for KV / larger batches).

## Measured — BF16 lever matrix (2026-08-25, same 1024/256 seed-42 workload)

Roofline check: peak BW ≈ 1792 GB/s (14001 MHz GDDR7 × 512-bit); BF16 27B = 54 GB/token →
theoretical max ~33 tok/s, ~28 realistic. Baseline 28.1 tok/s = ~85% of roofline (near-maxed).
Single-stream can only be beaten by fewer bytes/token (quant), more tokens/read (MTP), or more BW (TP2).

| Config | c1 tok/s | c1 TPOT | c1 TTFT | c8 tok/s | MTP accept |
|---|---|---|---|---|---|
| BF16 TP1 (baseline) | 28.1 | 35.0 ms | 188 ms | 196.9 | — |
| BF16 TP2 | 44.8 | 21.1 ms | 245 ms | 261.0 | — |
| BF16 TP1 + MTP(k3) | 51.0 | 18.1 ms | 200 ms | 298.1 | len 2.37 / 45.7% |
| BF16 TP1 + dyn-FP8 (`--quantization fp8`) | 47.8 | 20.3 ms | 124 ms | **325.7** | — |
| BF16 TP2 + MTP(k3) | **64.7** | **12.5 ms** | 223 ms | 315.3 | len 2.54 / 51.4% |

- MTP acceptance is low here because random-token input is unpredictable; expect higher on real text/code.
- dyn-FP8 needs zero download, halves VRAM, best batch throughput. Stacks with MTP/TP2 (matrix 2).

## Measured — combined winners + NVFP4 (matrix 2, 2026-08-25)

| Config | c1 tok/s | ×base | c1 TPOT | c8 tok/s | MTP accept | VRAM gpu0 |
|---|---|---|---|---|---|---|
| **fp8-dyn TP1 + MTP** | 75.3 | 2.7× | 10.1 ms | **458.0** | len 2.83 / 61.2% | ~82 GB (0.85 util) |
| **fp8-dyn TP2 + MTP** | **103.6** | **3.7×** | 7.8 ms | 439.9 | len 2.67 / 55.7% | ~82 GB/GPU |
| nvfp4 TP1 / TP1+MTP / TP2+MTP | — | — | — | — | — | FAILED to start |

**NVFP4 (unsloth/Qwen3.8-27B-NVFP4, W4A4, 22 GB) does not run with our flag set on SM120:**
`ValueError: backend FLASH_ATTN not valid... Reason: ['kv_cache_dtype not supported']` — the NVFP4
checkpoint requests an fp8/fp4 KV dtype that FLASH_ATTN rejects. Fix to try: drop
`--attention-backend FLASH_ATTN` and let vLLM auto-pick (likely FlashInfer). A foreground retry
without the forced backend got killed by the harness before yielding a result — **parked**. Given
FP8+MTP already delivers 2.7–3.7×, NVFP4 (lower W4A4 quality, flaky on SM120) is not worth chasing now.

## Recommendation
- **Daily driver (single GPU, leaves GPU1 free):** dynamic FP8 + MTP → **75 tok/s single / 458 batch**,
  ~half VRAM, zero download. Flags: `--quantization fp8 --speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.
- **Max single-stream latency (both GPUs):** FP8 + MTP + TP=2 → **104 tok/s**, TPOT 7.8 ms.
- MTP acceptance rises to ~55–61% here vs ~46% on plain BF16; real code/text should be higher still.
- Optional future: offline llm-compressor **static FP8** checkpoint = same speed, slightly better
  accuracy than dynamic; revisit NVFP4 only if a Blackwell-compatible attention backend is confirmed.

