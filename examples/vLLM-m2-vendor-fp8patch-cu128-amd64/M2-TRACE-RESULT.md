# M2 real H200 trace result — EP8 is a real bottleneck, not the whole story

## Service status and source integrity

- Target: original single-worker, 8xH200 M2 deployment on `prd-infr-k8h200` in `adeo` namespace; model id `M2`, K2P-LLM FP8 weights loaded from original PVC downloader; 262144 context; EOS 250019; FP8 KV cache; original image/runtime. Node `/tmp` overlay filesystem used for all profiling, staging and extraction.
- Tuned kernel configuration selected by natively installed vLLM `get_moe_configs`: `/etc/m2-moe-tuned/E=24,N=1792,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json`. This correctly reflects 192 experts split over 8 EP ranks => 24 local experts, full intermediate 1792, hidden 6144, topk 8. Actual stock vLLM uses split_k=1 and Triton block FP8 W8A8.
- Current profile-generation pod: `j-7ec325bf-0033-4a97-a688-e8ac30334349-5796f85468-srbsf`, Deployment generation 14. Ready, normal model server, restart count 0. No additional start-profile or inference calls were used during analysis. Model weights/runtime/model paths never changed.
- Temporary profiling flags still instantiated after this capture. Collector explicitly stopped (HTTP200) and no active profile; profiler scaffolding MUST be removed by separately authorized environment-only pod recreate if retaining normal service. Original nonprofile tuned generation 13 is saved in `pre-profile-tuned-gen13.json`.
- A one-request profiling client bug (parse an empty successful HTTP 200 as JSON) was fixed without changing the engine. First start means the capture covers two prefill/scheduler invocations and 29 decode steps, not exactly one uninterrupted 32-output-token request. Large gap due Python/CPU trace output serialization is excluded from performance analysis.

## Actual trace facts

- Eight Chrome `.pt.trace.json` traces exist and are complete (no .tmp suffix). Each ~16.45 GB -> ~131.6GB decimal (~123 GiB), due `with_stack`, `record_shapes`, entire CPU and GPU instrumentation. Not suitable for full `json.load()` in memory.
- All eight completed local byte-stream reductions with 0 malformed kernel object across 65,467-65,468 identified kernels/rank; every raw byte scanned, only GPU `cat=kernel` objects parsed. Tiny reports saved at `trace-export/trace-rank-table.json`. Rank0 markers at `trace-export/rank0-trace-boundaries.json`; rank0 compact kernel data at `trace-export/rank0-kernel-analysis.json`. Summaries and script preserved separately. Original 16GB traces kept temporarily per profile work, not exported over the relay nor duplicated to workspace.
- **Extractor limitation:** initial streaming extraction missed a kernel event whose boundaries straddled a 64MiB read chunk. Example rank0 extracted 3595/3596 expert GEMM calls; all vendor compact tables still retain exact 3596 calls / 76.639ms. No kernel event was invented or imputed to fix aggregate rows; the raw duration bias is less than one kernel (often tens of microseconds), but the reusable scanner is fixed to retain such incomplete prefixes. Original exported tables' raw-sum numbers are authoritative unless another costly eight-rank extraction is specifically needed.
- `gpu_user_annotation` events mark 2 prefill (`execute_new_1_cached_0`): 23.480666 + 18.607400 = 42.088066 ms; 29 steady decode (`execute_new_0_cached_1`): 395.810959 ms, median 13.647158 ms, min 13.274164 ms, max 14.361591 ms. There is 17.6ms between warm initial decode tail and next prefill, and an outer 93-second Chrome record span including profile startup and export; these are not included in decode totals. Profiler changes absolute timing and cannot be used as unp profiled HTTP throughput measurement.
- Interval-union all streams rank0: 337.657777 ms kernel active over 470.019947 ms outer kernel minimum-to-max => 132.362170 ms of gaps. Do not sum eight ranks as one-worker's latency; do not interpret nested `execute_new` self-CUDA duration 395.811 ms as sequential GPU time or service latency.

### GPU duration totals, per priority category

All are ms across the entire 2-prefill+29-decode section on each GPU/rank. Union takes overlapping launches out of total where streams overlap. These numbers are raw durations; not predictions of net end-to-end speedup.

| Rank | All kernel busy interval | Interval with no active kernel | Routed expert FP8 GEMM | Route/group/quant/etc. | Communication | Attention+RoPE | Other GEMM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 337.658 | 132.362 | 76.622 | 37.933 | 90.766 | 30.239 | 56.564 |
| 1 | 335.130 | 134.205 | 58.064 | 37.792 | 109.066 | 29.230 | 56.234 |
| 2 | 337.861 | 131.878 | 53.911 | 37.765 | 116.576 | 28.953 | 56.186 |
| 3 | 340.835 | 128.677 | 51.069 | 37.807 | 121.635 | 28.838 | 56.218 |
| 4 | 341.788 | 128.063 | 48.603 | 37.599 | 126.809 | 28.628 | 55.834 |
| 5 | 335.565 | 133.987 | 47.115 | 37.733 | 121.637 | 28.669 | 56.218 |
| 6 | 346.155 | 123.734 | 46.208 | 37.203 | 133.900 | 28.291 | 56.184 |
| 7 | 341.352 | 128.804 | 45.991 | 37.059 | 129.373 | 28.290 | 56.293 |

## What the trace proves about EP8

1. **Large expert-rank imbalance**. Rank0 routed expert kernel work is 76.639ms on the compact table (76.622 after one chunk-split extracted event was missed), while rank7 is 45.991 ms. All ranks have the same expected call count 3596 = 58 MoE layers x two GEMMs x31 forwards. It is a 1.6667x summed expert-time skew, ~30.65ms total across 31 forwards, averaging ~0.99ms/forward when including prefill. Exact decode-only attribution requires per-step timestamps for all ranks; this total remains unambiguous imbalance evidence.
2. **Inverse communication-time trend is synchronization pressure**, not healthy faster communication on a less-loaded rank: rank0 custom allreduce 49.284 vs rank7 84.285 ms. These kernels actively wait/collect until all ranks reach the barrier; rank with fast local expert GEMM spends more time spinning/participating in collectives. Expert loads cannot be summed with local expert time to claim duplicate work or exact wait time, because some real communication work and other output-reduction operators overlap.
3. **Constant MoE per-step overhead is substantial**: ~37-38ms for grouping, quantize, align, index etc. across 31 iterations. Two `per_token_group_quant` calls per MLP (3596) individually are only 8.54ms rank0. Do not say quantization alone costs 38ms: grouping/topk 12.26ms, reductions 8.66ms, per-token quant 8.54ms and small route/align operations all included.
4. **No dedicated all-to-all** appears for this DP1 stock `CompressedTensorsW8A8Fp8MoEMethod` path. EP ranks participate via fused local expert layout and existing Xllm tensor-parallel reduction path. Therefore it's incorrect to say "EP8 spends N ms in DeepEP all-to-all"; there is no such kernel. `ncclDevKernel_AllGather_RING_LL` = 3813 = 3 x 61 layers x31 iterative calls exactly, primarily two Xllm query-key partial RoPE allgathers and `ParallelLMHead` gather per layer. It must not all be charged to EP8.
5. Per-layer custom one-stage BF16 reduce 1-stage = 5611 (minus source's first uncached call for some shapes) corresponds to attention o_proj, shared expert down_proj, routed expert FusedMoE output each layer, not exclusively routed expert output. Each has group world_size 8; rank communication waits still includes actual work of attention/shared experts that removing expert parallel does not eliminate.

## Consequence (and what not to infer)

- Yes EP8-related rank imbalance and cross-rank synchronization is significant at this low batch (M=1). Small local tiles and all-to-all-ish rank imbalance turn useful GPU work into waiting in ordinary TP output reduction. Preliminary synthetic file tuning helped those kernels, but it cannot fix collective waiting.
- Not all comm is EP. TP o_proj, shared MLP down_proj and Q/K RoPE AllGather/lm_head cost still remain with or without expert parallel.
- Disabling `--enable-expert-parallel` while retaining TP8 means a DIFFERENT native expert sharding layout: 192 local experts with intermediate width `1792/8=224` each vs EP8 24 local experts with full width 1792. That is *not* removing all expert sharding and replicating all experts unpartitioned. Current tuned config E24,N1792 will not match this new shape. DO NOT copy/rename it to E192,N224; synthetic H200 tuning must be done against actual TP8-shape expert kernel.
- The next legitimate test is *temporarily remove EP only* with completely unmodified stock loader, same image/PVC/checkpoint and full acceptance plus direct warm-loopback fixed 256-output-token benchmark. To attribute effect, interleave with same nonprofile EP8 tuned baseline before permanently choosing. Old default-config throughput comparisons from mixed gateway are not valid as counterfactual.
- Equal or better TP8-only compute may be decided by small N=224 efficiency vs losses EP routing/skew; can't promise faster or 2x speedup without measured A/B. Since current trace proves imbalance, a small TP8-expert tiled synthetic experiment is evidence-based next step, not blind removal.

## Deferred cleanup / plan

1. Preserve compact all-rank reports (done); originals still `/tmp/m2-decode-profile-1`, only the diagnostic scripts and compact summaries are expected workspace changes.
2. No running profiler exporter or raw scan diagnostic remains aside from bounded single-process analysis. After copy/validation, raw ~123GiB may be deleted from `/tmp` with approval if not needed for further question. Release no original model/PVC/image weights.
3. Restore only exact original pre-profile tuned template (remove 8 `VLLM_*PROFILER*` and stack flags) in maintenance window after explicit consent. This requires full pod recreate + 15 minutes original recompile/warm loads. Preserve original tuned kernel file and tuned env. No source or hot patching whatsoever.
4. If user wants controlled A/B, tune actual N224/E192 shapes via rented H200-process temporarily within maintenance window then choose shape file per stock `get_moe_configs`, save weighted logs. Another generation needed to test TP8-only model layout. Keep rollback to exact original tuned EP8 template, do not make it permanent solely on profile.
5. Route load-balancing: `use_grouped_topk=False` on this stock M2, redundant experts=0. `EPLB` can't be silently introduced as supported for this quant kernel. Do not enable it simply because the raw rank imbalance exists; confirm method support for COMP vendor class first.

## Files

- `m2_trace_reduce.py`: bounded full-byte file scanner that parses only the `cat=kernel` records. Tiny result includes per-class and top-kernel records and per-stream union. Written/executed rank0/all-rank once each; `SyntaxWarning invalid decimal literal` cosmetic formatting, parser commit still validated via py_compile.
- `m2_trace_boundaries.py`: selects only two worker-iteration annotation names plus profiler record window markers. All events total 62 (31 pairs) plus outer markers, compact.
- `m2_stream_trace.py`: early unfinished byte filter/prototype, never executed, DO NOT use.
- `m2_analyze_profile.py`: originally intended whole-JSON reader, unsafe for 16GB trace; do not execute on full original files.
- `m2_trace_gate.c`: rejected unsafe `_exit` CUPTI preload experiment, never staged, never compiled/copied/executed. Keep unused; no injection, no signals and no runtime model edits.
