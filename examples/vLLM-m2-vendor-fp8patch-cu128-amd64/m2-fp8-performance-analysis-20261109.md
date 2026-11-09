# M2 FP8 performance/capacity diagnosis — 2026-11-09

## Direct answer

1. **KV token capacity is 4,779,392 tokens**, confirmed in the live engine by `kv_cache_utils.py:1286: GPU KV cache size: 4,779,392 tokens`. This is the **logical engine total** across all requests, not a per-GPU number to multiply again by 8.
2. It is backed by **298,712 blocks ×16 tokens/block**, with `cache_dtype=fp8_e4m3` and `enable_prefix_caching=True`. `Available KV cache memory: 69.51 GiB` is per GPU.
3. It is **not CPU-quota starved in observed tests**. Live `cpu.max=1600000 100000` =16 cores, `nr_throttled=0` and `throttled_usec=0` after >18k accounting periods. Fresh per-worker CPU deltas were roughly0.1–0.2 CPU core per GPU worker during request sampling (approx1.2 total cores), below the16-core quota.
4. FP8 compression does **not imply 2×** generation speed. Kernels and parallel code paths changed; a historical old-BF16 measurement is not an equivalent A/B.

## Exact memory/capacity interpretation

Per token, each GPU has one TP8-sharded KV head: 8 global KV heads /8 GPUs = 1. The model has `head_dim=128`, 61 layers and two cache entries K/V. At1-byte fp8 E4M3 that is:

$$61 \times 1 \times 128 \times 2 \times 1 = 15,616\text{ bytes of KV per token per GPU}$$

69.51GiB divided by 15,616 bytes gives ~4.78 million tokens, matching the logged pool. The identical logical sequence memory is present across the 8 tensor/expert ranks. Do not call the aggregate capacity eight times the published4.779M.

At BF16 cache the same bytes per token are twice as large; purely for a fixed cache memory budget this would be:

$$4,779,392 / 2 = 2,389,696 \text{ tokens}$$

That is a **capacity-only hypothetical**, not a measurement of an alternative BF16 deployment. BF16 weights consume more GPU memory, so the actual new-BF16-KV pool need not be exactly this value. A same-model `--kv-cache-dtype auto` probe is required to measure it.

One sequence has max length **262,144 tokens**. An idealized full-context packing is:

$$\lfloor 4,779,392 / 262,144 \rfloor = 18\text{ full-length sequences}$$

The actual simultaneous request cap is also influenced by `--max-num-seqs 256`, scheduling, prefix blocks and arbitrary context choices. It is not a guarantee of 18 concurrent full-context jobs or fast generation at that length.

## Why speed is not 40 ×2 =80 tokens/s

### Different kernel route

- Current compressed-tensors block-FP8 MoE method is `CompressedTensorsW8A8Fp8MoEMethod`.
- It selects `TritonOrDeepGemmExperts`. Direct module probe: `deep_gemm=False`, so the execution is the Triton route.
- `self.use_cutlass = not self.block_quant` means **[128,128] block-FP8 does not use the SM90 CUTLASS option.** The BF16-only `VLLM_USE_FLASHINFER_MOE_FP16` cannot magically enable an optimized FP8 MoE path.
- vLLM explicitly warns `Using default MoE config` because no file exists for `E=24,N=1792,NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128]`. Generic kernel tiling/tuning can be the bottleneck.

### Only part of every request is quantized computation

The original checkpoint deliberately ignores486 layers/components: attention, embeddings etc. remain outside the compressed expert tensors. Mixed precision, attention, gating, logits, norm, RoPE, and host-side decode/parser work remain.

This MoE uses top-8 of192 global experts, 61 layers. At batch=1 many GEMMs/groups are tiny; on-activity utilization of each H200 measured~66–71% in the samples with SM clocks 1980MHz and memory ~126253 MiB per rank. `nvidia-smi` utilization does not show individual kernel time or wasted slots, so it is a workload signal, not a complete SM-efficiency decomposition.

### Parallel/collective overhead remains

- TP8 attention normally uses reductions, EP8 expert dispatch/combination across8 GPUs despite compressed weight bytes.
- Exact vendor `XllmAttention.forward()` additionally all-gathers Q/K across TP, permutes them, applies RoPE and splits them because `rope_head_dim=64 != head_dim=128`.
- These communication/synchronization steps are not halved simply because expert weights are FP8.

CPU stays underquota and is not throttled, but CPU tokens processing, scheduling, collective submission, and GPU idle gaps between kernels can still cost latency even underquota. Distinguishing GPU compute vs NCCL/comm gaps requires a real NSYS/CUPTI/profiler trace, not inference from `nvidia-smi` alone.

## Numeric comparison scope

- External single-stream: **46.9–48.4 tokens/s externally**, +warm measured decode.
- Direct inside the same process environment: **512 tokens / 9.247s = 55.37 tokens/s**. This ~15% difference from the48.4 external value is consistent with gateway/path overhead but is different prompt/run/warm regime, not a formal latency decomposition.
- Small 8-stream warm-case: **2048 tokens / 4.601s =445 aggregate tokens/s (~56 each)**. FP8 is still useful for memory footprint and aggregate capacity, not just per-request latency.

The old `Model A` workload (former `j-e3a73157...` pointer) has **no live matching Pods** in the source runtime inspection. Historical40 tok/s therefore is not an apples-to-apples comparison, and this analysis does not claim the old path used an optimal BF16 kernel in the measured run either.

## Safe next profiling path (not initiated here)

- Capture a **same prompt/output length/time-outstanding** single-stream BF16 vs FP8 same SAME vendor image A/B with FP8KV vs BF16KV isolated.
- Bind the current FP8 128x128 kernel config for H200 E24,N1792,M=1 using the vendor's profiling tools; then compare before/after. No new image/source change has been made by diagnosis.
- For exact time split, sample CUPTI kernels, NCCL calls and CUDA gaps over a generation run in an isolated rehearsal, not by turning up CPU or production load.
- Prefer encoder-and-expert stable config first, not ad-hoc1→TP changes that break the block/expert dimensions or sacrifice shared memory cache.

Conclusion: **Memory savings are real. CPU quota is not limiting the measured path. Kernel selection and parallel/synchronization overheads are the credible bottlenecks; 2× single-stream performance is not warranted by FP8 alone.**
