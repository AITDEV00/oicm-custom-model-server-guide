# M2 bounded profiling — audit and authorized checkpoint-preserving trace

Read-only audit then explicit user-authorized profiler-enabled startup (generation14).
No code or weight replacement. The plan/planning JSON preserve the exact original
generation13 baseline and allow only profile env additions on main container.

## Why no trace on the already-running worker

At preflight, `VLLM_TORCH_PROFILER_DIR` was absent; the original eight rank
workers had profiler=None. The original vLLM had no `/start_profile` endpoint.
Attempts to profile that exact state are blocked, not a missing capture artifact.

## Actual authorized capture

The approved patch preserves the current checkpoint, vendor image/adapter, TP8/EP8,
EOS, TP8-node resources and validated native `E=24,N=1792` configuration. Only
profile settings and the trace dir beneath `/tmp` are added. The generated state is
new Worker UIDs, old worker gracefully terminated, no model re-download.

The vendor TorchProfilerWrapper is in `/opt/vllm/vllm/profiler/gpu_profiler.py`,
created by the imported GPUWorker class. It uses torch CPU+CUDA profiler activities
and on_trace_ready exports one trace per rank. `worker.profile()` dispatches
through normal engine collective requests. Python worker profiler.step() runs **at
worker iteration entry**, `VLLM_PROFILER_MAX_ITERS=32` plus the explicit stop bounds
decode capture but does not limit captured bytes. Full stacks and shapes may yield
large JSON; capture a short request after warm-up.

The new configured trace path is `/tmp/m2-decode-profile-1`; no PVC trace data.
No arbitrary LD_PRELOAD or CUPTI injection is used. The original unprofiled worker
is not hot-patched. `/tmp/m2_capture_profile.py` is an external stdin-staged HTTP
client not imported or read by the engine.

`profile(is_start=True)` in the live worker simply raises `Profiling is not
enabled` when `self.profiler is None`. There is no zero-restart torch-API route.
nsys is not installed. ncu profiles instrumented kernels and needs new process/RPC
launch support—it cannot attach a system trace to the fully loaded worker.

## Controlled startup actually applied under explicit user authorization

Retains the generation-13 exact Deployment except adds only eight main-container
profile envs: profiler dir, delay=0, max steps=32, shapes=true, stack=true,
profile-memory=false, disable-async-llm=true, gzip=false. `profile-activation-plan.json`
and `pre-profile-tuned-gen13.json` retain actual before/guard evidence.

Keep native `VLLM_TUNED_CONFIG_FOLDER=/etc/m2-moe-tuned`, stock image/adapter,
UID/immutable checkpoint and downloader/PVC/EOS exact. `/tmp/m2-decode-profile-1`
is in the main container filesystem, shared with rank subprocesses. There is no
PVC trace mount, code overlay or source image, and no CUDA/NVPROF injected load.

`/tmp/m2-decode-profile-1` must not be chmoded from a new init process until after
pre-creating it—it may already be read-only for nonroot userns mapping. In current
runtime node's normal main-container process-user mapping `/tmp` is writable.

Capture only after `Application startup complete`, CUDA-graph warm-up and one
small chat request; never while shard loading/compile is in progress.

## Capture procedure

Use the staged `/tmp/m2_capture_profile.py`, no PYTHONPATH override is needed:
it uses urllib only in a separate client subprocess. Make an unprofiled warm chat
request, call `/start_profile`, one 32-token same-prompt decode, `/stop_profile`,
then wait for all **8 Chrome trace JSONs + 8 profiler text summaries** to have
stable sizes for several seconds. There is no API-server trace here:
`VLLM_TORCH_PROFILER_DISABLE_ASYNC_LLM=1` excludes async_llm CPU traces by design.

```sh
mkdir -p /tmp/m2-decode-profile-1
export VLLM_TORCH_PROFILER_DIR=/tmp/m2-decode-profile-1
python3 /tmp/m2_capture_profile.py \
 --trace-dir /tmp/m2-decode-profile-1 \
 --output /tmp/m2-profile-1/report.json
```

No ops have run. If `--concurrency 8` is wanted, that is a separate new profile
window (one small burst) and should choose a different trace dir/report. Profile
window includes JSON HTTP parsing time. It is for attribution, not acceptance
throughput. Do not compare it directly to previously measured warm57.44rate.

## Analysis

Each rank plus API trace needs its own wall-clock normalization. From traceEvents
parse `ts`,`dur`,`cat`,`name`,`args`. No hand-summing GPU CPU self durations is valid.

Rank0 (`pid/tid` identified by process name/markers) is one GPU timeline. All rank
TP0 plus TP1..TP7 processes are distinct driver contexts. Do not add rank GPU time
and claim end-to-end latency: parallel GPU duration is duplicated eight times.

On one rank isolate:
- periodic CUDA graph replay / host launch events and associated stack;
- fused_moe_kernel (M1/2/4/8 config, fused expert compute);
- moe_align_block_size sorts, permute/routing gate, FP8 quant activations;
- NCCL comm kernels (`ncclDevKernel_AllReduce`, others)
- FlashAttention (`flash_fwd...`), DKV/linear GEMMs (`ampere/hopper/cutlass/...`),
  Elementwise/rmsnorm_RoPE/Gated topk kernels.

Build contiguous kernel busy intervals on CUDA streams; don't simply sum NCCL
if it overlaps compute on a separate stream. Idle = whole wall span minus busy
union (excluding inter-step gaps between profile start and request boundary).
Collectives can be busy time but their semantics alone still do not identify EP
correlation; native xllm TP reductions occur on every MoE layer, plus TP attn.
Label dominant compute vs comm vs wait. A top generic GEMM must not become an
MoE-local claim without full kernel name and observed stack/order.

To isolate EP8, next A/B would remove `--enable-expert-parallel` on a separate
controlled activation preserving TP8, image/PVC and EOS. That would use
`E=192,N=224,K=6144` (local experts192, expert width TP8), NOT the current
`E=24,N=1792` tuned file. It must run default selection/appropriate tile validation;
the pre-existing file will legitimately not match. Total experts and aggregate
model weights partition across8 ranks, not duplicate full192 experts.

A nonmatching `KEEP_EP` vs `TP_ONLY` paired profile (same prompt/cap/warm step)
quantifies effect; HTTP A/B cannot attribute to an EP all-to-all since DP1 generic
fused path does not dispatch/combine through specialized all-to-all. It is the
normal TP output reduction plus sharding/GEMM routing/dense alignment trajectory.

## Debug drafts not executed

`m2_trace_gate.c` was a rejected experimental CUPTI injection idea. It was never
staged or compiled; it must never be executed (its drain route would terminate
one rank). No injection, signal, nsys attach or profiler endpoint was issued.

`.start_profile()` API legality future next-start remains conditioned on flag
registered env. A failed endpoint under the current runtime is expected, not
measurement. No code was copied into producing M2. No new model/source image.
