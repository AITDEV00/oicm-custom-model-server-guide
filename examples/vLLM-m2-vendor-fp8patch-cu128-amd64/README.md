# M2 vendor FP8 diagnostics — final fix needs NO source overlay

The exact vendor image, immutable model/PVC/downloader and read-only license are
preserved. **Configuration generation 10** removed the experimental adapter
volume and mounted source. No derived image is needed or published.

## Actual root cause

The failed partial `--hf-overrides` supplied only `config_groups`, replacing
(rather than merging with) the original compression configuration. That omits
`quant_method` and selects `quantization=None`. Factory instrumentation proved
BF16 FusedMoE parameters without FP8 scale slots, causing the missing
`w2_weight_scale` error later in sparse weight handling.

The repair preserves the complete original quantization dict, removing only
the two null `scale_dtype` and `zp_dtype` fields per weights/input group for
compressed-tensors 0.12.2. In particular, all 486 original ignore entries,
`quant_method`, format/status and kv-cache metadata remain intact. The fixed
runtime constructs `CompressedTensorsConfig`. `m2-fp8-failure-20261109.txt`
contains the measurements, exact distinction from FP8 KV cache, and scope.

## Stock-vendor FP8 regression

`m2-fp8-stock-vendor-regression.py` uses the real compressed-tensors MoE
factory: FP8 weights, block-scale parameters, 24 local experts and a 192-entry
expert map. It passes the **unmodified vendor** `XllmModel.load_weights` for
ranks 3/4/7, placing gate/up/down weights and scales correctly for expert 100
(rank 4, local slot 4). Remote ranks return no changed weights, as expected.
Saved output is `m2-fp8-stock-vendor-check-20261109.txt`.

The normal sparse-per-expert group is retained until standard mapping. Global
expert IDs are translated by FusedMoE. An unowned tensor returning False with
`return_success=True` is a normal remote-skip branch, not an odd-input fatal.

## Historical experiments, not to build

`xllm_fp8_expert_mapping.patch`, `apply_patch.py`, `Dockerfile` and
`test_xllm_mapping.py` document r1 experiments. That overlay rejected a valid
remote expert, was removed from the Deployment and is **not** a recommended
fix. r3 was investigated but is unnecessary after the stock test above.
Real shard-loading and `/v1/models`/inference succeeded separately and are not
inferred from the factory test. See `FINAL-STATUS.md` and the captured actual
responses in `m2-fp8-default-eos-validation-20261109.json`.

## Native MoE tuning, profiler, EP/no-EP and n-gram conclusions

- **Use**: `MOE-TUNING-STRATEGY.md`, `M2-TRACE-RESULT.md`, `M2-NO-EP-RESULT.md`,
  `M2-SPECULATIVE-DECODING.md`, `M2-NGRAM-K1-K2-RESULT.md`, with validated
  sources (`m2_sweep.py`, `m2_moe_core.py`, `m2_moe_run.py`,
  `m2_trace_reduce.py`, `m2_trace_boundaries.py`) and compact JSON summaries.
  The small native tile file is made for the actual EP8 shape E=24,N=1792 and
  stock H200 loader; checkpoint weights and vendor image remain immutable.
- **Do not disable EP** in TP8 with the original FP8 checkpoint: routing MLP
  partitions width 224 and static block_n 128, so worker construction fails.
  No-EP throughput is unavailable, not zero or a throughput regression.
- TP-only invalid draft `m2_tp8_noep_experiment.py` is intentionally BLOCKED.
  It compiles as documentation of the exact error, but cannot be used to replay
  the invalid experiment. Do not rename E24,N1792 file to trick code paths.
- **Current retained EP8 state** is n-gram k=1 (min lookup 3 / max 5). Real
  warm candidate captures selected k1 instead of k2: k1 single median 75.734
  generated tokens/s (k2 71.263); final retained k1 75.769 and one warm
  8-stream aggregate 554.281 generated tok/s. See individual pod-bound source JSON.
- Warm loopback vs old warm no-spec performance cannot be wholly attributed to
  n-gram alone; a separate no-spec A/B was explicitly rescinded by the user.
- `m2_ngram_rollout.py` fast-winner keep/controlled experiment guards are
  resolved for this experiment only. `--keep-k 1` is valid from completed
  Generation19 k2 to k1; after Generation20 it does not repeat or replace.
- `--start-profile` ended after the one bounded capture; no profiling env
  remains on the final service. Eight-big traces are NOT committed. Very large
  original Chrome traces were reduced to restricted report JSON only.
- `m2_stream_trace.py` (incomplete), `m2_trace_gate.c` (rejected unsafe CUPTI
  /exit exp), `m2_moe_worker.py` and `m2_moe_tune.py` (incomplete prototypes)
  are intentionally local-only/ignored, not operational commands. Full
  provenance is in the relevant markdown reports.
- No credentials are saved; snapshot `envFrom` secret/configMap references by
  themselves are platform-managed identifiers like other operational config.
  Any subgroup overrides retain exact original quant, EOS and tile tuples.
- Debug relay expired after classic 4-hour sleep once; its replacement didn't
  change M2. Model-warm-up public-gate connection refused noise under 18000
  is not engine failure; reports distinguish from real worker exceptions.

## Commit-only metric scopes

Three warm serial requests + one burst is not a statistical recommendation for
all workloads; accepted drafts are recorded and verified not forecast. Always
compare same fixed inputs and _generation tokens/sec_, not thinking-text only.
