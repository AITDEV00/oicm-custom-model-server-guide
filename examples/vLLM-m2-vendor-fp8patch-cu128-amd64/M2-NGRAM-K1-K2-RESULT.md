# EP8 M2 n-gram k1/k2 measured selection (2026-11-10)

User instruction after the k2 candidate rollout: **do not restore the original no-n-gram EP8 configuration; retain the n-gram candidate with highest decode speed**. This supersedes the original speculative-off rollback plan.

Selection metric: directly loopback to `18000`, original single TP8+EP8 eight-H200 M2 engine, fixed same prompt, temperature 0, each response generation limit 256 tokens, three warm serial results; do not claim end-to-end gateway/network performance from this result. No eights-GPU concurrent engines, no checkpoint/image/PVC change, no profiler tracing.

## Measured candidates

| Original TP8+EP8 n-gram config | Fixed 26-token prompt single median generated tok/s | Repetitive prompt single median tok/s | Fixed 2-stream aggregate | Fixed 8-stream aggregate |
|---|---:|---:|---:|---:|
| k1 candidate, lookup min=3 max=5 | **75.7342** | **82.2056** | **149.5709** | **553.3867** |
| k2 candidate, lookup min=3 max=5 | 71.2629 | 81.0420 | 147.8136 | 521.7191 |
| **Final retained k1 on fresh pod, Generation20** | **75.7691** | **82.1726** | **157.1810** | **554.2814** |
| Historical no-spec prior warm direct protocol | 58.2424 median | not recorded | not recorded | 547.3696 median |

**k1 won all measured within-candidate criteria** (both single-stream fixed and repeating workloads; concurrency8 reference). It will be left enabled. k2's increased average acceptance (longer chains) does not overcome its longer target-verification cost in this workload/runtime. No counter acceptance alone is proof speed wins.

K1 serial fixed reference was the median of three warm runs; its 8-stream figure 553.3867 tok/s is one warm burst verified by the immediate completed token count, not a median of three independent long runs. New worker reload can add warmup/cache/parallel-capture ordering variance, so don't present these bounded candidate results as formal confidence intervals or every arbitrary production-workload behavior. Final Generation20 k1 direct-capture JSON is recorded separately (`k1-retained-final.json` after health passes).

## Correctness and n-gram observability

Both candidate workers were node-pinned `prd-infr-k8h200`, ready with 0 restarts. Each used original M2 image digest, complete original compressed-tensors quantization metadata/ignore list, FP8 KV cache, EOS250019, model id `M2`, context262144, original downloader/license/CPU-memory-8GPU config and `VLLM_TUNED_CONFIG_FOLDER=/etc/m2-moe-tuned` native E24/N1792 file. `--enable-expert-parallel` and TP8 remained on at all candidate restarts. No EP-off retry or weights/padding workaround exists.

- K1 engine accepted config from actual core boot; native H200 tuned config present. Model loaded 45.6663 GiB/GPU from 79/79 original shards before queries.
- Both API capture reports passed `M2 works` low-reasoning native EOS `stop` and `get_weather("Abu Dhabi")` `tool_calls` check.
- Same input at temperature0 deterministic identical output hashes/counters across K1 serial reference runs.
- Fixed three K1 serial requests: 19 draft tokens each, 10 accepted, mean acceptance chain 1.5263. Agg8: 152 drafts/87 accepted, mean chain=1.5724. Repetitive K1 singles: 52 drafts/29 accepted each; prompt still not maximally repetitive, don't generalize to arbitrary document tasks.
- K2 runtime counters had mean acceptance chain ~1.75-1.81 for fixed burst work, more than k1 but slower after verification. It is not selected just because acceptance is higher.
- Older reference no-spec and candidates were different periods/scopes and independent flags; candidate rates shouldn't be attributed purely to n-gram relative to older captures without a matched no-spec fresh comparison. The user explicitly canceled a new no-spec restore baseline, so this causal comparison was not performed. The direct candidate comparison is the requested speed decision.

## Safety problems handled during test (doesn't retread examples as baseline)

Baseline `k0-observed.json` and `baseline-ngram-harness-output.json` zero-byte artifacts were from a prematurely aborted old `/tmp` helper; explicitly excluded. No zero assignment suggests meaningfully no service. Independent through-stdin API request errors (`python - <helper>` argv vs stdin serializer) were fixed by self-contained fixed-source benchmark in `m2_ngram_benchmark.py`, verified from local SHA256 at execution inside the sole pod, particular UID / no private token. Capture output is written only on fully valid report parse+same pod identity.

Old container helper had natural 600-token trivial prompt ending in length during reasoning and contradictory reasoning assertion. Full corrected low-reasoning protocol included directly in the current helper and self-containedbench. This is not native EOS regression and not part of the k decision.

The ad-debug relay itself expired when its 4-hour sleep completed (exit0), and was recreated identical spec/host mount default-sa without changing M2. Repeated cluster access failed at shell errors once, not M2 workers. Raw readiness connect refused traces while private port 18000 not up were transient before readiness; these were explicitly distinguished from actual engine/worker fatal startup errors. No profiling traces needed/created.

## Exact selected Deployment (Generation20)

Desired state at user instruction:

```
TP8 + EP8 + ngram
--speculative-config {"method":"ngram","num_speculative_tokens":1,"prompt_lookup_min":3,"prompt_lookup_max":5}
```

Only the actual n-gram `num_speculative_tokens` changes between k2 candidate and chosen k1: original template normalizes equal after removing exactly the single `--speculative-config` two args. Retaining K1 has a scoped strategic main-args-array patch from the live healthy K2 template, not entire-original/zero-spec restore or historical profile env reinstatement. `m2_ngram_rollout.py --keep-k 1` now uses current main args for comparison, checks `generation19/k2`, fresh revision, ready UID, tested reports and normalized immutable baseline then applies only args value. `--restore` no longer exists.

**Generation20 final selected-k1 service is ready, tested, and kept running.** Final pod `j-7ec325bf-0033-4a97-a688-e8ac30334349-7474765585-49hfp`, UID `d0ec3b52-0eeb-4e27-8206-cb221928e29b`, has main Ready=true and zero restarts; Deployment has `readyReplicas=1` and `availableReplicas=1`. Original 79/79 shards and 45.6663 GiB/GPU weights loaded; native H200 E24/N1792 config selected; `Application startup complete`.

Actual live-pod `k1-retained-final.json` completed `all_checks_passed=true`: `M2` identity, 262144 context, native-EOS trivial low-reasoning probe (24 tokens, finish=stop), valid `get_weather('Abu Dhabi')` tool call, three warm fixed-reference serial requests (median **75.7691 generated tok/s**), repeat prompt **82.1726 tok/s**, fixed 2-stream aggregate **157.1810 tok/s**, and fixed 8-stream **554.2814 tok/s**. Captured standard burst gen-counter skew was zero for 1/2/8. Private identity/API and n-gram metrics confirmed. No profiler.

**No no-spec rollback occurred after the user's instruction.** Original checkpoint/image/PVC/license/EOS/FP8 override/EP8/tuned config remain in use. Selection complete; no additional n-gram restart or no-spec phase is pending. Final actual pod load boot was completed with native model warmup before final read-only capture succeeded.

## Artifacts

- `h200-moe-measurements/ngram-restored-lock` absent; no such restore step required.
- `final-k1-ready.json`, `final-k1-boot.log`, `final-deployment-live.json`: actual high-confidence final positive k1 cluster boot & identity proof.
- `k1-observed.json`, `k2-observed.json`: completed measurements bound to their separate exact pod UIDs.
- `k1-live-deployment.json`, `k2-live-deployment.json`: candidate-transient verified identities and templates.
- `phase-k1-plan.json`: final desired template and scoped directive preserving current k2 all-other-fields before k1 keep.
- `win-k1-direct-args-patch.json`: locally generated proof of exact depth-only change; scoped runtime operation passed immutable guards via main script.
- `k1-boot.log`, `k2-boot-current.log`, `k2-boot-and-first-attempt.log`: original worker load traces; read excluded gate connect refused as workers fatal.
- `m2_ngram_benchmark.py`, `m2_ngram_capture.py`, `m2_ngram_rollout.py`: current bounded tools, no profiler and no speculative-off step.
