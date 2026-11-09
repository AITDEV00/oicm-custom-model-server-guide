# M2 vendor runtime: speculative-decoding compatibility audit (2026-11-10)

## Verdict

- **N-gram: implemented and candidate for a controlled measured A/B.** It is the only lightweight speculation available without introducing new model weights. Do not claim faster decode until a matched warm-loopback A/B proves it.
- **Native MTP: not available for this M2 checkpoint/runtime.** `XllmForCausalLM` has no MTP draft-head architecture in this installed registry. Installed registered methods normalize DeepSeek, MiMo, GLM4-MoE, ERNIE, Qwen3-Next, LongCat-Flash and Pangu Ultra MoE-specific heads into `method: "mtp"`. No Xllm equivalent exists.
- `architectures: ["XllmForCausalLM"]` in the original checkpoint `config.json`, no draft/speculative/MTP configuration. Original `model.safetensors.index.json` has 67484 named tensors; zero mtp/draft/speculative/eagle/nextn/medusa matching names. No new checkpoint head should be fabricated from the last dense or MoE layer.
- Xllm has `SupportsEagle3` target-side interface (source `/opt/vllm/vllm/model_executor/models/xllm.py` class declaration at 831) but **the installed generic EAGLE3 guard explicitly only supports model types llama, qwen, minicpm, gpt_oss** (`config/speculative.py` 620–631), and there is no external trained EAGLE3 head on the PVC. Interface presence is not full support. Enabling EAGLE3 anyway would be rejected by the config guard even before the missing trained head. No runtime modification here.

AUDIT was **read-only**: no CLI/serve option, model weights, configuration, running environment or pod was changed. No model or second engine was loaded. `SpeculativeConfig(method='ngram', ...)` for k 1,2,3 was validated in-process only; model config was not initialized and no CUDA graphs were recompiled. No speculative decoding needs or suggests modifying M2 original absolute context or eos overrides.

## Validated installed implementation

Image is original pinned m2-vllm vendor 2.9+cu128 environment with /opt/vllm vendor code, evaluation interpreter `/opt/eval_xllm`. In this source:

- `/opt/vllm/vllm/v1/spec_decode/ngram_proposer.py` has actual Numba KMP `NgramProposer`, scans CPU-side context token ids, longest matching suffix and emits up to k existing-prefix continuation tokens for verification. `find_spec('numba')` verified installed.
- `/opt/vllm/vllm/v1/worker/gpu_model_runner.py` constructs NgramProposer when `speculative_config.method == "ngram"`. Only TP rank 0 performs CPU n-gram proposal at GPU proposer stage (`propose` call around line 3310). Target model still validates sampled chains normally across all eight TP ranks, preserving output distribution for supported speculative sampling. Other request parameters unsupported by speculation use bypass set in `gpu_input_batch.py`, checked via `/v1/spec_decode/utils.py`.
- Ngram numba uses one CPU thread here by design because TP8 divides `min(1, physical CPU count)` by TP. That CPU overhead is small for one/two tiny requests but main target-forward graph and inter-rank work scale with `uniform_decode_query_len = 1 + k` (gpu_model_runner 551).
- Original per-token `VLLM herd needs (1) GPU draft scores with lossless verification. Speculative tokens that fail are rejected/rolled back; batched target execution still has graph padding and model/attention scheduling work even when hit rate is low.

## Candidate safe settings (not applied)

Recommended first feasibility configuration:
```json
{
  "method": "ngram",
  "num_speculative_tokens": 2,
  "prompt_lookup_min": 3,
  "prompt_lookup_max": 5,
  "disable_by_batch_size": 8
}
```
This exact in-process construction succeeds. Keep k=1/2 as short bounded sweep; k=3 accepted config but likely progressively more compute. Do NOT first enable k=4+ or a synthetic learned MTP alternative.

`prompt_lookup_*`: ngram pattern matching caps, not M2 max-model-len/token length. Output may have zero available draft tokens and revert to one target-generated token per iteration naturally. Prefix-cache acceleration remains a distinct mechanism. Low temperature/highly repetitive chat meaningful test.

If enabling `disable_by_batch_size=8`, verify actual scheduler telemetry in controlled low concurrency then concurrency>8 before relying on batch-count gating. Source stored config/validator exists, but no unconditional use of this symbol was found outside `config/speculative.py` in vendored source at audit; batch-size cutoff must not be represented as proven code-path enforcement until loading the controlled engine and exercising it.

## Metrics / matched A/B

- Warm all standard non-profile service configuration first; remove current profiler env only in authorized maintenance window. Do not benchmark while the stack+shape profiler exports giant /tmp files.
- Fixed local loopback model requests, same user identity path, model id M2, fixed temperature (include temperature zero and realistic sampling separately), no external gateway comparison with unrelated mixtures. Record generated completion token count, median/p95 of at least 3 warm repetitions, aggregate throughput at 1, 2, 8 streams under fixed serving max batch. Warm JIT n-gram first.
- Make baseline/shape/no-target same: original image, EP8 tuned `/etc/m2-moe-tuned/E24,N1792..., KFp8, 262144/eos 250019/tool choice preserve as applicable. No interference with currently saved trace or K2 H200 spare worker.
- Read per-pos `num_accepted_tokens`/`num_draft_tokens` Prometheus labels from exact installed `v1/spec_decode/metrics.py` metric helpers (classes count real draft sets plus rejected position arrays; `SpecDecodingStats.observe_draft`). Also log human accepted-rate output before/after warmup. Don't use per-iteration historical stats as instantaneous speed.
- Compare native tokens/sec direct loopback and per-request wall time, not user-visible stream gaps alone. For at most three single-stream workload types: reasoning prose (expected low), exact constrained repeating content/code (possible high), long prompt then continuation. Both baseline and n-gram on same warm engine.
- Always maintain readiness/no error and exact tool-call/eos smoke acceptance. ngram changes scheduler behavior and token positions; even if numerical outputs should match, verify tool parsing and multi-format reasoning parser. Non-greedy outputs cannot demand exact bytes if hash seed graph order changes randomness semantics; acceptance should use forced repeat with temperature 0 and reasoning/tool behavior checks separately.

### Ceiling measured from prior real trace

Captured two prefill plus 29 decode GPU fragments under vendor FP8 TP8/EP8 (independent of ngram M query) had expert fused ranks 46–77ms, route/group/quant support ~37ms per rank, custom reduction durations 49–84ms, attention RoPE ~28–30ms, other linear ~56ms. These existing sync+support costs remain with speculation. Target verification k>1 could increase GEMM expert batch and change load balancing beneficially only if accepted, but in worst case it repeats full work for discarded tokens. Actual local tuned expert results local M1/M2 aren't a service-wide gain guarantee; true $r = \mathbb{E}[\mathrm{accepted}+1]/(\mathrm{per-iteration target+draft+tracking})$ needs measuring, not claiming blanket 2×.

## Conclusion

The best next speculative experiment is bounded **ngram k=1 and k=2** with acceptance-rate observability in a non-profile configuration, not MTP. Best action by default is **ngram disabled** until hit rate and steady decode comparison justifies it. Nothing was enabled from this audit.

## Follow-up: TP8-only experiment is impossible in this FP8 runtime

User requested remove-EP first then n-gram only if improvement. Actual no-EP TP8 worker failed before loading weights because the stock vendor FP8 MLP gate/up width per partition 1792/8=224 violates `weight quantization block_n = 128`. No valid no-EP service/benchmark exists; n-gram cannot be enabled on a no-EP engine as planned.

Recovery applied cleanly: standard EP8 tuned config, no profiler flags, no n-gram. Exact saved pre-profile EP8 tuned template was first restored; explicit JSON replacement of the entire `containers/0/env` path was verified to remove the profile list (the previous strategic merge returned no-change). Wait for recovered EP8 service readiness and acceptance; no further speculative edits.

Evidence `h200-moe-measurements/tp8-noep-failure-boot.log` records actual root ValueError and `h200-moe-measurements/tp8-noep-rollback.json` records the pre-profile EP8 original recovery spec.
