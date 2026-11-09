# M2 FP8 — final measured resolution (2026-11-09)

## Serving acceptance: PASS on exact vendor runtime

- Deployment `j-7ec325bf-0033-4a97-a688-e8ac30334349`: generation **11**, Ready 1 / available 1.
- Pod `...-c76f57d85-5zxft`: Ready, **zero restarts**, eight H200 with TP8/EP8.
- Exact original vendor image digest; stock adapter no volume overlay. Adapter SHA256 `30c1b7bcf19365c4db9e826da4f11d1da5f11fa39057fedbc3ac454a2ab1a9af`.
- Full **79/79** safetensors checkpoint loads, CompressedTensors FP8 MoE, compilation, KV allocation and CUDA graph capture succeed.
- Exact original source/download contract, PVC and read-only license preserved. No model weight/config file changes or re-download.

## Final required configuration

1. Preserve **complete original quantization metadata** (`quant_method`, `ignore`, format/status, scheme, related fields); remove only unsupported `scale_dtype:null` and `zp_dtype:null` under each weights/input group. All **486 ignore entries** remain. Current partial metadata constructed *unquantized BF16* experts lacking FP8 scale slots, which caused the original `w2_weight_scale` KeyError. It was not a valid local/global expert-renumbering issue.
2. Array of sampling defaults: `--override-generation-config {"eos_token_id":250019}`. The template's `<|im_end|>` marker is ID250019 but checkpoint config/generation defaults use `<|endoftext|>` ID1. Correcting this **in memory** prevents ordinary chat from remaining inside reasoning until max_tokens.
3. Serve name **M2**, max length **262144**, FP8 KV-cache and TP8/EP8 exactly preserved.

Actual full-config runtime constructed `CompressedTensorsConfig`; a standalone exploratory query used an incompatible hand-written prefix and is not valid regression evidence. A real FP8 factory instantiation and stock-vendor load callback regression is retained with numerical correctness for ranks 3/4/7.

## Actual inference verification, no stop override sent by the client

- `GET /v1/models` → HTTP200, only model ID `M2`, `max_model_len=262144`.
- Normal chat → HTTP200, `finish_reason=stop`, response text `M2 works` (236 output tokens including reasoning).
- Tool prompt requiring direct invocation with low-reasoning chat template → HTTP200, `finish_reason=tool_calls`, parsed `get_weather` with `{"location":"Abu Dhabi"}` (25 output tokens).
- A too-short 220-token default high-reasoning request can stop at `length` before its final response. This is expected reasoning budget behavior; it is distinct from the corrected EOS and FP8 issues.

## External gateway and small performance baseline

The new deployment gateway `https://inference.ai.ecouncil.ae/models/7ec325bf-0033-4a97-a688-e8ac30334349/proxy/v1/` was verified with real HTTP200 M2 requests. Exact M2 identity and user-supplied secret were exercised but are not stored in evidence files.

A small warm benchmark (see `m2-fp8-throughput-20261109.md`) measured **48.4 external tokens/s one request** and repeated eight-way short-prompt rates of around **434–445 aggregate tokens/s**. The confirmed final batch generated 2048 tokens in4.601 seconds = **445.1 aggregate / ~56 per request**, with the engine generation counter advancing by exactly2048. A second independent eight-way 384-token burst took6.80–7.02 sec/request (~440 aggregate). The asynchronous 10-second logger snapshots report196.8–204.8 lower-window aggregates when requests straddle their boundaries; that is not a contradiction. No deployment change was required and the Worker remains Ready with zero restarts.

## Scope

Repeated-prompt/warm-cache results are not a full load performance or cold-start certification. Long/extrapolated-context quality, saturation load and accuracy still require separate benchmarking. No source-overlay image should be built/published: stock source regression PASSED and the historical overlay files are for diagnosis only.

See `m2-fp8-final-deployment-20261109.json`,
`m2-fp8-final-immutable-adapter-sha256.txt`, `m2-fp8-stock-vendor-check-20261109.txt`,
`m2-fp8-default-eos-validation-20261109.json`, and the corrected
`m2-fp8-failure-20261109.txt` for captured measurements.
