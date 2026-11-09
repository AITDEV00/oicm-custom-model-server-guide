# TP8-without-EP experiment: unsupported by M2’s existing FP8 checkpoint loader (2026-11-10)

## Decision under test

User plan: first TP8 with `--enable-expert-parallel` removed and no speculative decoding, compare performance; only if improved second restart with n-gram. This was authorized and attempted, not a hypothetical speedup projection.

Original M2 identity/PVC/image/262144/EOS/kvFP8/runtime are preserved throughout. During no-EP failed worker, the model did not complete weight-loading and never served, so no benchmark or n-gram was added.

## Root failure, proved from full previous worker log

Target identical M2 image at `m2-vllm@sha256:d3ced94ef...90209ba9`, TP8 fixed, noEP argument, full complete compressed-tensors hf_overrides preserved from the working FP8 configuration, same `VLLM_TUNED_CONFIG_FOLDER=/etc/m2-moe-tuned` EP-tuned file untouched. Native code computes:

- EP8: each rank has `24/192` physically local experts, each full `intermediate_size=1792` width (valid fp8 blocks).
- No EP TP8: each rank has 192 local experts, each gated/up output `1792/8=224`, down input similarly 224.

Actual **first worker exception** at 21:27:30:

```
ValueError: The output_size of gate's and up's weight = 224
is not divisible by weight quantization block_n = 128.
```

This originates during `XllmForCausalLM` model instantiation/worker init, before `79/79` weight-shard loading. Repeated engine/core initialization messages later are just wrappers, not causal resource failure. Original previous log copied to `h200-moe-measurements/tp8-noep-failure-boot.log`; no profiler or inference errors were the cause.

Consequence: no supported no-EP TP8-only baseline exists in this original FP8 runtime. EP group labels can still print TPn/EPn but that is not evidence '--enable-expert-parallel' remained live; FusedMoE's `use_ep` dependent local expert shapes make the hard distinction. No renaming of E24/N1792 tile file, editing/checkpoint padding/replacing quantizer/weight magic or kernel/source is permitted to conjure a pass.

## Recovery, including exact profiler removal

The later misnamed local snapshot `pre-tp8-noep-generation14.json` was renamed to `observed-tp8-noep-generation15.json`: it actually contains the failing no-EP config, not authoritative working baseline. Authoritative safe restore is only `pre-profile-tuned-gen13.json`.

Saved generated rollback patch `h200-moe-measurements/tp8-noep-rollback.json` from known-good full exact `pre-profile-tuned-gen13.json`. Whole template strategic patch restored TP8 EP8 but profile env remained as part of generated controller reconciliation/rollout generation15→16, so that service health is not claimed as non-profile.

To make profile removal explicit and atomic, live env verified exactly the eight expected `act.ENV` profiler keys plus K8S/K8S_NAMESPACE/VLLM_TUNED_CONFIG_FOLDER, then used a JSON `"replace"` operation only for `/spec/template/spec/containers/0/env` with `K8S_POD`, `K8S_NAMESPACE` and `VLLM_TUNED_CONFIG_FOLDER=/etc/m2-moe-tuned`. This returned `deployment ... patched` with generation17 and retained original EP8/tuned/no-other-resource fields. Earlier strategic env rewrite returned patched(no change) and wasn't adequate to prove active state cleanup; don't repeat that same method as a claiming non-profile restore.

Generation17 returned the original clean hash `84fc97ff`, final pod `j-7ec325bf-0033-4a97-a688-e8ac30334349-84fc97ff-mwv8t`. The uncertain transient profile-enabled reconciliation attribution is not recast as proven external action.

**Recovery complete and valid.** Worker Ready=true, restarts=0, no profiler environment/events/traces, no speculative config. All 79/79 original checkpoint shards loaded; original H200 FP8 identity `/M2`, `max_model_len=262144`, EPS/TP original flags and immutable image/PVC/quantization context are intact. Confirmed `Application startup complete`; boot log explicitly selected the existing H200 E24/N1792 config through `VLLM_TUNED_CONFIG_FOLDER`.

The first ordinary natural 600-token trivial-answer check did not stop: `finish_reason=length` after lengthy default reasoning. This is not proof of missing EOS or a throughput/EP regression. An isolated original `chat_template_kwargs: {"reasoning_effort":"low"}` check completed **24 output tokens** with content exactly containing `M2 works`, `finish_reason=stop`. Low-reasoning tool call completed `get_weather({"location":"Abu Dhabi"})`, `finish_reason=tool_calls`. The standard acceptance helper now invokes the existing known-good low-reasoning protocol for the initial trivial chat too; tool probe already used it.

Saved EOS/tool record: `restored-ep8-low-reasoning-tool-20261110.json`. Recovery completion no longer depends on the generic 600-token natural-reasoning acceptance check which deliberately had no effort override.

`m2_tp8_noep_experiment.py` is documentation only with a blocked-replay RuntimeError; its known `from __future__` syntax needs removal/reordering if ever imported again, but such import/replay is unnecessary and unsafe for this concluded experiment. `m2_profile_activation.py` has obsolete generation guards and must not be invoked against final Generation17. Do not repeat the pre-noEP restore approach without changing its expected current-state guards.

## Performance / decision

| Configuration | Result |
|---|---|
| Previous TP8+EP8 tuned warm private benchmark | single ~57.17–57.77 (median ~57.44), 8-stream ~547.54–548.57 aggregate tok/s (median ~548.20) |
| **Restored Generation17, same fixed-protocol TP8+EP8 tuned test** | single 58.239/58.242/58.553, **median 58.242**; 8-stream 546.255/547.370/548.262, **median 547.370 aggregate tok/s**. Three burst token deltas verified engine-side **2048** each; direct exact original prompt with 26 input tokens and exactly 256 generated/output tokens per request, two warmups first, same `temperature=0`, no profiler/n-gram active |
| TP8-without-EP | startup FAIL output_size 224 not divisible 128, **no throughput available (not zero)** |
| TP8-without-EP + n-gram | **not executed**, first-step service never loaded; n-gram cannot solve weight construction |

The recovered generation performs close to the prior small warm reference: single median is within about +1.4%, aggregate median about -0.15%; this is ordinary small-run variation, not a claimed improvement from any new change. Direct-loopback warm reference is not a gateway round-trip comparison and excludes queue/network distance.

No claim EP disabling improves speed. Further alternatives require a supported compatible FP8 TP-only quantization path or different genuine layout/runtime (requires user approval and own checksum/validation), not an environment-only flip.

The known EP8 nondeterministic rank skew remains a valid optimization target. N-gram k1/k2 is only a potential separate *EP8* experiment, with independent user decision and explicit restart authorization. **No such experimental speculative test was added in the recovery.** It must not be interpreted as satisfying the original removeEP-first plan. Reuse the saved post-restore baseline if that separately approved variant is ever tested.

Performance JSON: `h200-moe-measurements/restored-ep8-warm-performance-20261110.json`.
