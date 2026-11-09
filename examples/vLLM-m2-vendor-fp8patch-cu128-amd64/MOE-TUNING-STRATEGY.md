# M2 FP8 MoE tuning — exact vendor audit and one-rollout result

Status (2026-11-09): **accepted. One explicitly authorized original-M2
restoration was deliberately applied at generation 13. Current Deployment is
ready/available 1/1; current worker `j-7ec325bf-0033-4a97-a688-e8ac30334349-84fc97ff-fgvxs`
is Running 1/1 on the original H200 node with main restart count 0. The production
log confirms `79/79` shards loaded and vendor selection of the mounted exact
`E=24,N=1792,NVIDIA_H200,fp8_w8a8,[128,128]` native JSON with zero target
`Using default MoE config` warnings. API startup completed normally with no
OOM or engine-core initialization failure. `/v1/models` retains model ID M2,
the original root and `max_model_len=262144`.

Functional requests returned HTTP 200. Normal chat omitted any stop override,
ended with `finish_reason=stop`, returned M2 works using native EOS behavior,
and kept K2v3 reasoning separate from content. A direct tool invocation parsed
to `get_weather(location=Abu Dhabi)` with `finish_reason=tool_calls`. EOS override
remains `{"eos_token_id":250019}` and image/checkpoint/adapter configuration remains
unchanged. Raw responses plus final runtime evidence are stored under
`h200-moe-measurements/`; see `m2-restored-acceptance-summary.json`.

Warm direct loopback measurements used the original 26-token prompt, deterministic
sampling, 256 generated tokens per call, two warm-up requests, three sequential
single-request samples and three identical eight-stream batches. Generation-token
server counters advanced exactly 2048 tokens in each eight-stream measurement.
Medians were 57.44 generated tokens/s single-stream and 548.57 aggregate tokens/s
for eight streams. These exclude public-gateway overhead; while historical
external values were 48.4 and 445.1 respectively, these are not matched-path
interleaved A/B tests. They are observations under the stated conditions, not
an isolated attribution of all observed change to the tuned artifact, not a
global full-model speedup guarantee, and not evidence for 2x. No additional
rollout was needed or performed.**

The preserved pre-maintenance Deployment was generation 11 with corresponding
UID `cbe870c3-263f-4460-a6ae-05b9d4fb9b76`. Its source checkpoint/PVC,
downloader/license/full quantization override/pinned image and 262144/EOS
behavior were kept unchanged. Final main-container change is ONLY the
read-only native JSON mount plus `VLLM_TUNED_CONFIG_FOLDER`, not a new model or
engine source overlay.

Completed exact-shape synthetic evidence: representative 49 configurations per
M, 32 emitted power-of-two-count mode trials per bucket plus eight valid
out-of-memory candidate exclusions at each of M1/2/4/8; total 160 valid and
32 explicitly rejected over-shared-memory configurations. Three paired
finalist/default 128-iteration executions at each M passed tensor numerical
sanity checks. Local-expert speedup factors relative to actual vendor default
are: M1 1.105766, M2 1.156253, M4 1.086483 and M8 1.110764.
**These are local-expert GEMM measurements, not full model throughput.** The
original true default includes the extra `SPLIT_K=1` key at runtime.

Native file selected by the actual vendor loader:
`E=24,N=1792,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json`
with exported 637-byte SHA256 `82ba001d4da159284fc0ed788dcfa9f1a9b0afa8c577a1e390f71e2e1df052d5`.
The in-pod ConfigMap version has no trailing newline but the same parsed content;
its 636-byte SHA256 is `3bae3deebf03f95297b31c11800bfe462920753d12ad7f0eac691bc16ccf4c81`.
Evidence, raw checkpoint and finalist timing files are saved under
`h200-moe-measurements/`. The independent one-H200 helper was deleted only after
those files were hash-verified and saved outside its ephemeral volume. No proof
of an exhaustive global tile optimum or of a 2x full-M2 throughput increase exists.

The rest of this document preserves the original research/capacity plan for future
maintenance; cluster observational sections below are historical, not live status.

## What the exact runtime supports

The pinned LLM360/vllm `v0.12.0-ifm` image (run revision ce42a74a8, package
`0.1.dev11870+gce42a74a8.d20260422.cu128`) already contains
`/opt/vllm/benchmarks/kernels/benchmark_moe.py`, Ray and Triton 3.5.0. The tuner
uses locally synthesized local-expert tensors, CUDA graphs and the exact in-image
Triton kernel. It does **not** need to load the 379 B model or download duplicate weights.

The correct external-artifact variable for this release is
`VLLM_TUNED_CONFIG_FOLDER=<directory>`. Do not use the suggested
`VLLM_FUSED_MOE_CONFIG_FILE`: that string is not present anywhere in the exact image.
`get_moe_configs` first tries `<folder>/<exact-filename>` and then packaged configs.
Environment and parsed config lookups are cached after initialization; no supported
hot-reload or `get_moe_configs.cache_clear()` entrypoint exists. Activation of an
additional artifact therefore requires new worker processes; it cannot promise zero
restarts.

Current requested shape, byte-identical output filename:

```
E=24,N=1792,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
```

It is **not** E192: TP8+EP8 means each rank has 192/8=24 local experts, complete
per rank, so expert TP=1. The local weight shapes are:

```
w13 = [24, 2*1792, 6144]
w2  = [24, 6144, 1792]
```

The loader derives N from w2.shape[2]=1792. There is **no exact matching artifact**
among the configs shipped in this image (no `E=24,N=1792*` at all); do not borrow one
from a different E/N/device/dtype/block shape. With no DeepGEMM installed, the current
block-FP8 route really falls back to Triton. `VLLM_USE_FLASHINFER_MOE_FP16` applies
to unquantized BF16 and is not an FP8 optimization.

## Why a direct copied command is unsafe here

The bundled old tuner uses these fields generically:

- `intermediate_size` (M2: **16384**) whereas the actual routed width is
  `moe_intermediate_size` (**1792**).
- `num_local_experts` (absent) whereas M2 has `num_experts=192` and top-k=8.
- Top-level `quantization_config.weight_block_size` (absent) whereas its actual
  block shape is `config_groups.group_0.weights.block_structure=[128,128]`.

Running the generic online command against M2's current model would therefore
select the wrong internal shape or fail instead of tuning the serving kernel.
Supply a **small tuner-only metadata directory/config.json**, derived by read-only
mapping from the original checkpoint, using the vendor's corresponding architecture
family only for the shape parser: `Qwen3MoeForCausalLM`, `num_experts=192`,
`num_experts_per_tok=8`, `moe_intermediate_size=1792`, `hidden_size=6144`,
`dtype=bfloat16`, and `quantization_config.weight_block_size=[128,128]`.
Keep `block_structure=[128,128]` and activations `dynamic=true`, grouped by 128 in
the evidence mapping. Add only standard Transformers classification fields (for
example `model_type=qwen3_moe`, `vocab_size=32000`) so that the off-the-shelf
same-version HF parser can load the synthetic **metadata**, not any M2 code. This
does not assert that M2 is Qwen3 or changes its Xllm architecture: it reuses only
the tuner's python-if channel that consumes exactly the four MoE dimensions. The
real matching key is w13/w2 dimensions above. This directory must never replace
the model directory, checkpoint, service HF overrides or vendor Xllm adapter.
It is only the tuner's input description; the benchmark tensor contents are synthetic.

The vendor tuner divides experts by `--tp-size` in its EP branch and does not divide      
intermediate width. Therefore the correct synthetic geometry is
`--tp-size 8 --enable-expert-parallel --dtype fp8_w8a8` on one physical H200.
`--tp-size 8` describes deployment partition geometry; it does **not** allocate
eight GPUs to that process. Do not drop EP or run `--tp-size 1`: those change
the requested E/N shape rather than the production TP8 topology.

Source references:
- [Exact pinned loader](https://github.com/LLM360/vllm/blob/v0.12.0-ifm/vllm/model_executor/layers/fused_moe/fused_moe.py#L816-L891)
- [Exact pinned tuner, search/save/input](https://github.com/LLM360/vllm/blob/v0.12.0-ifm/benchmarks/kernels/benchmark_moe.py#L186-L267)
- [Exact tuner filename helper and old block-size behavior](https://github.com/LLM360/vllm/blob/v0.12.0-ifm/benchmarks/kernels/benchmark_moe.py#L535-L564)
- [Exact source Triton fallback](https://github.com/LLM360/vllm/blob/v0.12.0-ifm/vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py#L122-L156)
- [Exact env variable registration](https://github.com/LLM360/vllm/blob/v0.12.0-ifm/vllm/envs.py#L1447-L1448)
- [Upstream maintainer supports keeping tuner in container + targeted batch sizes](https://github.com/vllm-project/vllm/issues/28456#issuecomment-3545002683)
- [Official profiling guide](https://docs.vllm.ai/en/latest/contributing/profiling.html) (choose Nsight Systems for low overhead; profiler adds overhead, don't benchmark while tracing)

## Fewest-restart procedure

1. **Prepare once, without production changes:** recognize exact pinned-image and H200;
   extract only the original config to the tuner-only metadata directory; preflight
   the derived E/N/K/block shape expected above. No source code is mounted into M2.
2. **Reserve a spare H200 for a separate tuner worker.** Scheduling `nvidia.com/h200: 1`
   will not select one in-use half GPU. It requires an actually free allocatable device.
   Use the same image/CUDA to avoid incompatible compiled-kernel choices. Give its Ray
   temporary directory scratch storage, and isolate resources to avoid touching M2.
   Check available host capacity rather than assuming a free H200 slice exists.
3. **Baseline + sweep inside the tuner** using `benchmark_moe.py` with an explicit
   `--batch-size 1 2 4 8`, then moderate in-flight sizes appropriate to actual
   service load (e.g. 16 32 64 128), never by starting eight model replicas. Test
   several random seeds and always compare with its real current default
   `{BLOCK_M:64, BLOCK_N:128, BLOCK_K:128, GROUP_M:32, warps:4, stages:3}`.
   The old search rejects BLOCK_N/K smaller than quant blocks, leaving
   5 M-tiles * 2 N-tiles * 2 K-tiles * 2 warp choices * 4 group choices
   * 4 stage choices = **640 configurations for each M**. BLOCK_K=256 is clamped
   to 128 by this kernel, so deduplicate it and its otherwise identical neighbors
   to reduce up to half the sweep. The offline script itself must show print
   `args`, resulting `block_quant_shape=[128,128]` and initial E/N/K before
   starting any sweep; a wrong top-level metadata file is a preflight failure.
   Do not automatically inherit newer code or a "special M2 kernel" from upstream
   main. Targeted sizes avoid the old script's default **18-size/640-per-size**
   sweep. Its native output is only saved at the end; prefer a tiny offline
   wrapper with per-M checkpoint/resume and winner output safe to kill, not
   repeated production loads.
4. **Validate A/B offline on same GPU + deterministic seeds:** time multiple best/default
   interleaved runs and require numerical tolerance comparing outputs of chosen tile
   and original default on identical synthesized inputs/weights/gating. Speed alone
   is not an acceptance gate; do not accept NaN/Inf, wrong scale indexing or missing
   E=24,N=1792 filename. Include its Triton version metadata and image digest as inputs.
5. **Write the winning JSON once into its own read-only ConfigMap** with only six well-known
   keys per M plus Triton version; hash it, and test vendor loader returns integer-M
   maps and the exact path is used. `VLLM_TUNED_CONFIG_FOLDER` must be an
   environment variable (or part of the environment ConfigMap) on **main**, not
   only a name mounted under it. The path mounts at a new directory such as
   `/opt/m2-tuned-moe`, with the full artifact name above as the file. Preserve
   the entire existing download, downloader, license, PVC, full quantization
   override, tool parser, generated EOS, TP8/EP8 and FP8KV args. Do not replace
   the source config/PVC or use the model directory for tune results.
6. **Only after A/B acceptance, one normal worker restart/rollout (if final A/B needs
   a same-node one-eight-H200 rehearsal).** Current Deployment strategy is Recreate,
   so no claim of existing zero-downtime rollout is made. The candidate must log
   "Using configuration from <mounted path> for MoE layer" with matching missing
   warning absent; all ranks Ready,79/79 load, tool/chat acceptance with native
   EOS, then consistent identical prompt output-length concurrency gateway benchmarks.
7. **If tuner gain is insufficient or only in the synthetic metric, stop. No speed
   rollout is justified.** Next is trace-attributed bottleneck analysis across Triton,
   collectives and wait gaps, not blindly changing TP, FP8KV or new communications libs.

Two distinct options exist for full-size execution measurement:

- Preferred: isolated second full8-H200 candidate, using same downloaded PVC read-only,
  new profile-only evaluation endpoints/names. It gives an actual A/B before touching
  serving Deployment; it requires a new eight-card node and scheduling parity.
- Otherwise: one maintenance-window same-node Recreate yielding new workers once for
  the prevalidated artifact. Hold the previous ConfigMap/env state for rollback.
  The known model 79/79 download and compilation cost still applies; do not promise
  zero outage or a few-second restart. That validation outage is not multiple tune
  boot iterations—the kernel sweep happens before it.

## Current cluster capacity constraint (observed 2026-11-09)

Only `prd-infr-k8h200` exposes GPU capacity:8 H200/8 generic NVIDIA devices. The
current M2 already requests8/8. Other AD worker nodes advertise no H200. Therefore
**no spare H200 exists for isolated GPU tuning or a second full8-card candidate** at
this instant. A separate1-GPU pod will be Pending and must not be treated as isolated.
Do not shrink current GPU allocation or scale the model to zero just to tune without
authorization. Request a temporary spare H200 node/card or permit a maintenance window
for a non-serving tuner using free GPU capacity on that same node when M2 is offline.
"Offline" means model weights/engine are not reloaded per candidate, not magically
free access to all of H200 compute during full-serving workload.

Even with tuning achieving a large expert-kernel reduction, this addresses only the
two grouped-GEMM calls in an MoE layer. Gate quantization, routing, activation,
MoE sum, expert collectives, attention, remaining BF16 linear layers and scheduler
do not automatically double. End-to-end target improvement must be measured before
saying this solves 2x versus the old40 tok/s figure; a2x local GEMM ratio is at most
an upper bound, not predicted service throughput. When the synthetic artifact's
share improvement is material but total gain is small, no more generations should
be spent guessing. Isolate kernel/collective/CPU-wait portions under NCU/timeline;
`/start_profile` appears in this old runtime but cannot be used unless the profiler
output directory was enabled at worker init. Newer v0.13 profiler-config syntax
is not proof it applies here. NCU is bundled; nsys was not found.
