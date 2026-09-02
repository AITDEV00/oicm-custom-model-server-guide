# OICM vLLM image for VibeVoice ASR (AWQ int4) — vLLM 0.27.1 / cu129

Custom `oicm/vllm` serving image with **microsoft/VibeVoice** baked in as a
vLLM plugin, enabling the AWQ-int4 split-decoder export
`lemuriandezapada/VibeVoice-ASR-awq-int4` on **vLLM 0.27.1** +
**transformers 5.15** (both shipped in the `vllm-openai:v0.27.1-cu129` base).

## Why a custom image

Upstream VibeVoice's `vllm_plugin` targets vLLM 0.17.x / transformers 4.x and
fails on the 0.27.1 stack in three independent ways:

| # | Failure | Fix baked into this image |
|---|---------|---------------------------|
| 1 | `vllm_plugin/model.py` uses 0.17 multimodal APIs (`_get_data_parser`, old `ProcessorInputs(mm_data=...)`) | quant_compat patch: `get_data_parser` on `BaseProcessingInfo`, new `ProcessorInputs(mm_data_items=..., tokenization_kwargs=...)` |
| 2 | transformers 5.15 **natively registers** `vibevoice_acoustic_tokenizer` / `vibevoice_asr`; upstream's module-level `AutoModel.register(...)` raises `ValueError: already used by a Transformers model` at import time, killing the whole plugin | all 6 module-level registrations + the plugin's own registrations wrapped in `try/except ValueError` |
| 3 | `transformers.models.qwen2.tokenization_qwen2_fast` no longer exists (merged in transformers 5.x) | import shim falling back to the top-level `Qwen2TokenizerFast` |

Plus: attention-context alias fallback fixed for 0.27.1 (`virtual_engine` was
removed from `ForwardContext`). `diffusers` is installed because the
`vibevoice` package import chain requires it.

The full patch is preserved in `patches/vibevoice_vllm_0_27_transformers5_compat.patch`
(applied to upstream `microsoft/VibeVoice` @ `94da20d`, 2026-07-24).

## Build

Requires the base image `localhost/oicm/vllm:0.27.1-cu129` (built from
`../vLLM-podman-0.27.1-cu129-amd64/`).

```bash
# from this directory
podman build --platform=linux/amd64 \
  -t oicm/vllm:vibevoice-awq-0.27.1 \
  -f Dockerfile .
podman tag oicm/vllm:vibevoice-awq-0.27.1 localhost/oicm/vllm:vibevoice-awq-0.27.1-cu129

# or via the repo Makefile (from examples/)
make build-vllm-podman-vibevoice-awq
```

## Entrypoint (startup.sh)

The image ships its **own** `/app/startup.sh` (overrides the base image's),
with the same OICM contract plus VibeVoice-specific handling:

| Behavior | Detail |
|---|---|
| Split-decoder-aware discovery | The model dir has TWO `config.json`s (root + `decoder-awq/`). The base image's newest-`config.json` scan can pick the decoder. Here each candidate config is inspected: only one with `architectures: ["VibeVoice*"]` or a `vibevoice_metadata` key wins. |
| `MODEL_DOWNLOAD_FOLDER` | Optional explicit override (recommended for S3-pulled models with a known subpath). |
| `MODEL_ID` | Passed by OICM; defaults to `vibevoice`. |
| `INIT_CONTAINER_USE_PVC` / `PVC_PATH` / `USE_DATA_VOLUME` | Same as base image (S3/object-storage init container drops weights on the PVC). |
| Air-gap | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` enforced — nothing fetched at runtime. |
| AWQ defaults | Auto-injects `--trust-remote-code` (required) and `--dtype float16` (required for AWQ int4) unless the operator set them in `EXTRA_ARGS`. |
| WSL2 dev | `VLLM_WSL2_ENABLE_PIN_MEMORY=1` exported by default (no-op on cluster). |
| `PORT` | Server port via `PORT` env (default 8080), or just pass `--port` in `EXTRA_ARGS`. |
| `EXTRA_ARGS` | Normalized through the same `arg_normalizer.py` as the base image (=, space, JSON, dot-notation all accepted). |

## Run

```bash
# Local test (model on disk)
podman run -d --name vibevoice-asr \
  --device nvidia.com/gpu=all --security-opt label=disable \
  --network host \
  -v /srv/shared/download/hf/lemuriandezapada/VibeVoice-ASR-awq-int4:/pvc-home/model:ro,Z \
  -e INIT_CONTAINER_USE_PVC=True -e PVC_PATH=/pvc-home \
  -e MODEL_ID=vibevoice \
  -e EXTRA_ARGS="--max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.85 --no-enable-prefix-caching" \
  localhost/oicm/vllm:vibevoice-awq-0.27.1-cu129
```

For OICM (S3-pulled weights): leave `MODEL_DOWNLOAD_FOLDER` unset and the
discovery finds the VibeVoice root automatically; set it explicitly only if
multiple models share the PVC.

## Transcription endpoint shape

vLLM 0.27.1 does **not** register OpenAI's `/v1/audio/transcriptions` route
(404). VibeVoice is served as a **chat model with audio input** — use
`POST /v1/chat/completions` with an `audio_url` content part:

```jsonc
// Request
{
  "model": "vibevoice",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "audio_url",
       "audio_url": {"url": "data:audio/wav;base64,<b64>"}},
      {"type": "text", "text": "Transcribe this audio."}
    ]
  }],
  "max_tokens": 1600,
  "temperature": 0,
  "repetition_penalty": 1.05
}
```

The response is the **standard ChatCompletion schema** (`id`, `object`,
`choices[].message`, `usage`, ...) — no audio-specific wrapper. The
transcript itself is in `choices[0].message.content` as a JSON string array
of timestamped segments:

```json
[
  {"Start": 0.0, "End": 12.26, "Speaker": 0,
   "Content": "I can't believe you did it again. I waited for two hours..."},
  {"Start": 12.81, "End": 23.28, "Speaker": 1,
   "Content": "Look, I know I'm sorry, all right?..."}
]
```

> **Known quirk:** the model does not emit EOS after the audio ends — it keeps
> generating hallucinated segments with growing timestamps (e.g. up to 1600 s).
> The transcript up to the real audio duration is accurate (verified against
> `2p_argument.json` ground truth). Mitigate client-side: cap `max_tokens`
> (~1500 for ≤70 s audio), then discard segments whose `End` exceeds the audio
> duration. `rtf_benchmark.py` implements this filter.

## RTF benchmark (RTX 5090, AWQ int4, vLLM 0.27.1)

`RTF = processing_time / audio_duration` (lower is better; < 1 = faster than
real time). Measured with `rtf_benchmark.py` (3 runs/file, best-of, after
1 warmup, loop-cut params):

| Audio | Duration | Processing | RTF | Segments recovered |
|---|---:|---:|---:|---:|
| `2p_argument.wav` | 68.5 s | 7.0 s | **0.102** | 6 (all, matches GT) |
| `2p_argument_1_5x.wav` | 45.7 s | 6.9 s | **0.152** | 6 |
| `2p_argument_2x.wav` | 34.3 s | 7.5 s | **0.225** | 6 (matches GT) |

~10x real-time on the 1-minute clip. Reproduce:

```bash
python3 rtf_benchmark.py --base-url http://localhost:8321 \
  --audio-dir /home/jyao/ADEO/vibevoice.cpp/audio --runs 3 --warmup 1
```

Note: without the loop mitigation (plain `max_tokens=4096`, no filter) the
decode keeps running to the cap and RTF degrades to 0.26–0.52; the audio-time
processing above is the true transcription cost.

## Launch flags — what and why

The flags quoted on the model card (`--trust-remote-code`, `--dtype float16`,
"`prefer awq_marlin`") and the upstream launcher defaults were validated on
vLLM 0.27.1 / SM120 (RTX 5090):

| Flag | Decision | Why (verified) |
|---|---|---|
| `--trust-remote-code` | **Required** — auto-injected by `startup.sh` | custom `vibevoice` architecture is not in transformers |
| `--dtype float16` | **Required for AWQ int4** — auto-injected by `startup.sh` | AWQ quantized weights are fp16; bf16 would upcast and lose the quantization benefit |
| `--quantization awq_marlin` | **Do NOT set** | Engine log confirms auto-selection already picks Marlin: `quantization=auto_awq` → `Using MarlinLinearKernel for AutoAWQMarlinLinearMethod`. Forcing it would pin the backend and lose the auto path the model card explicitly recommends (`prefer letting vLLM infer the backend from config.json`). |
| `--no-enable-prefix-caching` | **Keep** (upstream default) | The plugin's `_call_hf_processor` adds a **random salt** to every audio item specifically "to ensure unique hash and bypass cache" (`model.py:851`). Prefix caching therefore can never hit. A/B measured (3 runs x 4 files): RTF 0.101–0.202 with caching OFF vs **identical** 0.101–0.201 with it ON — zero benefit, so disable to skip the bookkeeping. |
| `--enable-chunked-prefill` | Keep (upstream default) | Matches `vllm_plugin/scripts/start_server.py`. |
| `--max-model-len 32768` | sensible default | 65 536 also works but doubles reserved activation memory for no benefit on ≤1 h audio. |

Make targets (from `examples/`):

```bash
make build-vllm-podman-vibevoice-awq   # build image
make run-vllm-podman-vibevoice-awq     # run container on :8321
make benchmark-vibevoice-awq           # RTF benchmark vs running container
```

## Notes

- The split-decoder weights are loaded by the patched `VibeVoiceForCausalLM`
  (`vibevoice_decoder_model_path` / `vibevoice_decoder_quantization` in the
  root `config.json`); the repo layout must stay intact — `decoder-awq/` is
  part of the model.
- On H100/H200 (SM90) vLLM auto-promotes AWQ to `awq_marlin`; do not force
  `--quantization awq` explicitly.
- Keep `--max-model-len` at 65536 only if the GPU has headroom; the audio
  encoder consumes VRAM outside the KV cache.

## OICM deployment

Push to Harbor and reference like any other custom vLLM image. The entrypoint
is the image's own `/app/startup.sh` (same env contract as the base
`oicm/vllm:0.27.1` image: `INIT_CONTAINER_USE_PVC`, `PVC_PATH`, `MODEL_ID`,
`EXTRA_ARGS`). Only the generic args belong in **Model Server Arguments** —
`--trust-remote-code` and `--dtype float16` are injected automatically:

```
--max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.85 --no-enable-prefix-caching --enable-chunked-prefill
```
