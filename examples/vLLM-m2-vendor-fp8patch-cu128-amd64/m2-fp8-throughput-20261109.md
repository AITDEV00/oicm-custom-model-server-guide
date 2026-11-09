# M2 FP8 external gateway throughput — 2026-11-09

Target deployment: `7ec325bf-0033-4a97-a688-e8ac30334349`, served ID **M2**, original 8x H200 TP8/EP8 FP8 runtime.
Connection HTTPS gateway (TLS verification disabled for these tests) → existing license gate 8080 → private vLLM 18000. All tests returned HTTP200.
No actual deployment configuration, model files, license or PVC was changed during these tests. No credentials are persisted here.

## Latest measured runs

Identical 26-token prompt, temperature 0, completion capped at 256 tokens, warm engine with native M2 finalization. Results are **not cold-cache or long-context performance**.

| Streams | Actual per-request wall | Client tokens/s each | Client aggregate tokens/s | Mean engine TTFT |
|---:|---|---:|---:|---:|---:|
| 1 | 5.29s | 48.4 | 48.4 | 28.2ms |
| 2 | 4.05–4.06s | 63.0–63.3 | 126.0 | 28.9ms |
| 8 | 4.66–4.72s | 54.2–54.9 | 433.6 | 30.0ms |
| 8 confirmed repeat | 4.51–4.60s | 55.7–56.8 | **445.1** | not sampled precisely |

The confirmed repeat’s server `generation_tokens_total` advanced **8633→10681**, exactly 2048. Its wall-clock aggregate is **2048/4.601=445.1 tokens/s**, or ~55.7–56.8 per request. This is the cleanest final eight-way measurement. The earlier approximate per-request counter-to-decode ratios should not be misread as per-stream rates: counter-time accounting is asynchronous across overlapping engine steps.

### Eight-request validation burst

Second burst (fresh prompt, 384 tokens/request) completed **3072 tokens in 6.80–7.02 seconds/request**, only ~0.22s spread between fastest and slowest stream. Approximate eight-stream aggregate is **440.8 tokens/s**, bounded measured range **438.7–453.8 tokens/s** from fastest/slowest single request duration. Repeated prompts affect prefix-cache/prefill behavior; this is a warm benchmark.

### Logger evidence

- `18:31:55`: **196.8 tokens/s**, **8 running**, 0 waiting; this is vLLM’s first logarithm-window **aggregate**, not per-request speed. `18:32:05` became 110.4 tokens/s with 0 running; `18:32:15` 0 tokens/s/0 running after requests finished. For the final repeat the logger exported **204.8** tokens/s at `18:33:05` with running already0. The batch lasted4.6s but its log export straddles a 10-second window / idle boundary. Do not infer the burst peak from the incomplete log interval; use exact counters/wall timing and actual tokens for the bounded measured rate.
- `18:29:45`: 53.4 tokens/s, 1 running; `18:29:55`: 49.8 tokens/s, 2 running; `18:30:05`: 64.0 tokens/s, 0 running at completion. Those earlier sequences combine different prompt lengths and idle/prefill/capture diurnal timing; do not extrapolate a saturation curve.
- Time-to-first-token in recent releases of the measured calls was approx 25–33ms internal, not counting external TLS/auth/gateway/network latency. End-to-end first-response HTTP probe 5.03s, 236 output tokens, about 46.9 tokens/s.

### Interpretation/bounds

- This is a small warm benchmark, not a full load-sweep, and not a claim of sustained maximum M2 throughput.
- No queueing observed in these bursts; daytime cluster traffic is not isolated.
- Prefix caching changes TTFT for repeated prompts; number of streams/prompt lengths in logger and client windows can differ.
- Engine checkpoints remain generation11 Ready=1/zero restarts actual serving M2. This test run is non-mutating and no source patch was needed.

## Evidence

- Numeric counters and client timing: [m2-fp8-throughput-20261109.json](m2-fp8-throughput-20261109.json).
- Second burst/service log capture: [m2-fp8-engine-throughput-snapshot-20261109.json](m2-fp8-engine-throughput-snapshot-20261109.json).
- Recent exact vLLM log throughput summaries: [m2-fp8-throughput-engine-logs-20261109.txt](m2-fp8-throughput-engine-logs-20261109.txt).

Sensitive Bearer credentials intentionally not recorded; rotate any key copied into an external chat.
