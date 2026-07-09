# TP=1 vs TP=2 Benchmark — Qwen3.6-27B-FP8 on vLLM 0.24.0

**Date:** 2026-07-09
**Hardware:** 2x RTX PRO 6000 Blackwell Workstation (SM120, ~96GB each), PCIe (no NVLink)
**Software:** vLLM 0.24.0, torch 2.11.0+cu130, CUDA driver 595.58.03
**Model:** Qwen3.6-27B-FP8 (dense, hybrid GDN), served with the same flags as `start-qwen.sh`
(FLASH_ATTN backend, prefix caching, chunked prefill, no MTP, `--max-num-seqs 256`)

## Results

Workload: `vllm bench serve`, random dataset, 1024-token input / 256-token output, seed 42.

### Single stream (concurrency 1, 6 requests)

| Metric | TP=1 (1 GPU, 0.80 util) | TP=2 (2 GPUs, 0.85 util) |
|---|---|---|
| Decode (TPOT) | 19.8 ms/tok = **50.5 tok/s** | 13.0 ms/tok = **77.0 tok/s** |
| Median TTFT | 121 ms | 200 ms |

**TP=2 decodes +53% faster** despite all-reduce running over host shared memory
(P2P disabled on this machine).

### Concurrency 8 (48 requests)

| Metric | TP=1 | TP=2 |
|---|---|---|
| Output throughput | 324 tok/s | **369 tok/s** (+14%) |
| Decode (TPOT, median) | 21.2 ms | 16.4 ms |
| Median TTFT | 940 ms | 1442 ms |

## Conclusions

- **Dual GPU (TP=2) is the speed default** for the 27B: splitting the dense weight
  reads across two GPUs' memory bandwidth more than pays for the SHM all-reduce cost.
- **TP=1 wins only on TTFT under load** (prefill batches pay more for all-reduce)
  and when you need the other GPU free.

## vLLM 0.24.0 breakage found during this benchmark

Both vLLM sections of `start-qwen.sh` failed to start on 0.24.0 until fixed:

1. **Mamba cache limit (hybrid GDN models):** default `max_num_seqs=1024` exceeds
   available Mamba cache blocks (888 at 0.80 util, TP=1) and is now a hard startup
   error. Fix: `--max-num-seqs 256`.
2. **DeepGemm default-on for FP8:** crashes with `Unknown recipe` on this checkpoint
   (scale_fmt is not ue8m0 — same issue as SGLang's DeepGemm backend). Fix:
   `VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`.

Both fixes are applied in `start-qwen.sh` as of this date.
