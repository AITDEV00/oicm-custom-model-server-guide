# Drama 3 (Fish Audio) — H200 tuning reference

Practical, measured settings for serving **`fishaudio/drama-3`** with
SGLang-Omni. Everything here comes from the v0.2.0 cookbook, the model
`config.json`, and live tests against the Al Ain H200 deployment.

- **Image:** `registry.adeoaiengine.ecouncil.ae/openinnovationai/platform/mlops/mlops-serving/sglang-omni:0.2.0-cu130-amd64`
- **Example folder:** `examples/SGLang-podman-omni-dev-cu130-amd64/`
- **Build:** `make build-sglang-podman-omni-dev`
- **Deliver:** `make deliver-sglang-podman-omni-dev`

---

## 1. What the model is

Read directly from the checkpoint's `config.json`:

| Property | Value |
|---|---|
| `architectures` | `["FishOmniModel"]` |
| `model_type` | `fish_omni` |
| **Modality** | **Text → Speech. No audio input.** |
| Text backbone | 40 layers, hidden 2048 |
| MoE | 256 experts, **top-8**, `moe_intermediate_size` 512 |
| Layer pattern | `linear_attention` ×3 → `full_attention` ×1, repeating |
| Attention | `attn_output_gate: true`, `head_dim` 256 |
| Context | `max_position_embeddings` 262144 (engine caps at **16384**) |
| Quantization | `bsgemm-moe`, block [128,128], dynamic activation |
| Audio decoder | `codebook_mtp_shared`, **10 codebooks** |
| Audio encoder | `null` |
| Weights | 37.0 GB (fp8 experts + bf16 remainder) |

Output audio is **44.1 kHz mono 16-bit PCM** — *not* the 24 kHz OpenAI default.
Clients that assume 24 kHz will play back at the wrong pitch and speed.

---

## 2. Single H200: the short answer

All three stages (preprocessing, tts_engine, vocoder) default to GPU 0, with the
vocoder in its own process. **On one H200 the defaults are already the optimum
for most workloads — start with no flags at all.**

```bash
sgl-omni serve \
  --model-path /data-volume \
  --port 8080
```

Only add flags for the specific situations in section 3.

---

## 3. Flags by situation

### High concurrency (more than 16 simultaneous streams)

```bash
sgl-omni serve \
  --model-path /data-volume \
  --tts_engine.gpu_memory_fraction 0.85 \
  --tts_engine.factory.max_running_requests 256
```

> ⚠️ **With the engine at 0.85 on the GPU that also runs the vocoder,
> concurrent `stream: false` requests can run out of GPU memory.** The vocoder's
> frame budget is derived from `--vocoder.gpu_memory_fraction` and does not
> account for memory held outside PyTorch's allocator. If you raise the engine
> share, do one of:
> - serve those requests with `"stream": true`, or
> - keep the engine at **0.70** on the vocoder's GPU, or
> - move the vocoder to a second GPU.

### Two or more GPUs

```bash
sgl-omni serve \
  --model-path /data-volume \
  --tts_engine.gpu 0 \
  --vocoder.gpu 1
```

On one H200 this is not possible; on multi-GPU nodes it buys **~72% more
throughput and ~37% lower median TTFA at 256 streams** (Table 2 below).

### Long reference audio (over 60 s)

```bash
sgl-omni serve \
  --model-path /data-volume \
  --preprocessing.factory.max_reference_audio_seconds 90 \
  --preprocessing.total_reserve_bytes 9GiB
```

The preprocessing stage's memory scales with reference length and is capped by
`total_reserve_bytes` (default **6 GiB**, sized for the 60 s limit). **Raise the
reserve together with the limit** — a reference that does not fit fails with an
OOM inside preprocessing. The server checks at startup that all stage budgets
fit the card.

### Long prompts

A prompt is capped at **8192 rows** (template + input text + reference
transcripts, plus ~21.5 rows per second of reference audio).

```bash
--tts_engine.factory.max_prompt_rows 16384
```

A prompt plus its `max_new_tokens` must still fit the engine's context length
(16384).

### Same-GPU process sharing

```bash
--mps on
```

Lets engine and vocoder share one GPU via CUDA MPS. Can raise throughput under
high concurrency but **may make a single stream slower**. Benchmark before
keeping it.

---

## 4. Measured performance on one H200

Streaming requests, Seed-TTS English, one pre-encoded reference, `temperature`
1.0 / `top_p` 0.9 / `top_k` 30 / `max_new_tokens` 500.

### Table 1 — default placement, by concurrency

| Concurrency | Throughput (frames/s) | TTFA p50 | TTFA p95 | RTF mean |
|---:|---:|---:|---:|---:|
| 1 | 166.7 | **88.9 ms** | 90.8 ms | 0.131 |
| 2 | 287.0 | 96.2 ms | 130.1 ms | 0.152 |
| 4 | 467.6 | 108.2 ms | 156.4 ms | 0.186 |
| 8 | 678.1 | 153.2 ms | 210.9 ms | 0.256 |
| 16 | 927.6 | 210.1 ms | 287.2 ms | 0.373 |
| 32 | 1,244.7 | 304.6 ms | 396.1 ms | 0.551 |
| 64 | 1,687.3 | 450.9 ms | 1,020.1 ms | 0.801 |

A single stream gets first audio in **under 100 ms**. At concurrency 16 audio is
generated ~2.7× faster than realtime; at 64, mean RTF is still 0.80.

### Table 2 — vocoder placement at high concurrency

1,088 requests per level, mean of six server starts.

| Vocoder placement | Concurrency | Throughput (frames/s) | TTFA p50 | TTFA p95 |
|---|---:|---:|---:|---:|
| Engine's GPU | 128 | 2,177 | 685 ms | 1,412 ms |
| Engine's GPU | 256 | 2,422 | 3,403 ms | 4,255 ms |
| **Second GPU** | 128 | **3,692** | 401 ms | 995 ms |
| **Second GPU** | 256 | **4,160** | 2,146 ms | 2,469 ms |

With the vocoder on the engine's GPU at 256 streams it is fully loaded and each
new stream's first audio waits in its queue. Moving it off gives ~72% more
throughput and ~37% lower median TTFA.

### Table 3 — fp8 vs int4 checkpoint

1,088 Seed-TTS English prompts, concurrency 8, non-streaming.

| Checkpoint | Weights | Latency mean | p95 | RTF mean | req/s | WER |
|---|---:|---:|---:|---:|---:|---:|
| `fishaudio/drama-3` (fp8) | 37.0 GB | **1.235 s** | 1.902 s | 0.281 | **6.47** | 1.99% |
| `fishaudio/drama-3-int4` | 20.8 GB | 1.334 s | 2.178 s | 0.320 | 5.99 | 1.63% |

**int4 is ~8% slower** but uses 16 GB less VRAM and has slightly better WER. Only
take it when you need the memory headroom for KV cache. Requires compute
capability ≥ 8.0 (fine on H200).

---

## 5. Sampling defaults — leave them alone

| Parameter | Default | Notes |
|---|---|---|
| `temperature` | 1.0 | Per-request override supported |
| `top_p` | 0.9 | Per-request override supported |
| `top_k` | **no cap** on the main token | Codebook heads cap at 30 regardless |

> ⚠️ **Setting a request-level `top_k` of 30 on the main token lets a stream
> fall silent after its first phrase.** The codebook heads are already capped at
> 30; a wider cap there costs speaker similarity.

`repetition_penalty` **must be 1.0** — any other value is rejected with HTTP 400.

---

## 6. Request contract

### Plain TTS

```bash
curl -X POST http://host:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"fishaudio/drama-3","input":"Hello, how are you?","response_format":"wav"}' \
  -o out.wav
```

### Streaming

```bash
curl -N -X POST http://host:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"fishaudio/drama-3","input":"Long text here.","stream":true,"response_format":"pcm"}' \
  -o out.pcm
```

**`stream: true` requires `response_format: "pcm"`** — anything else returns
`400 stream=true requires response_format='pcm'`. Supported formats overall:
`wav, mp3, flac, pcm, aac, opus`.

Measured live on the Al Ain H200: TTFA **141 ms**, 74 chunks, 6.31 s of audio.

### Voice cloning

```json
{
  "input": "Get the trust fund to the bank early.",
  "references": [{
    "audio_path": "https://.../prompt.wav",
    "text": "We asked over twenty different people, and they all said it was his."
  }]
}
```

Each reference may be **at most 60 s** (see section 3 to raise it).

### Multi-speaker

Tags `<|speaker:0|>` … `<|speaker:15|>` — one text can carry up to 16 characters.
Only speaker `0` is supported for single-voice output; other speaker tags are
rejected.

---

## 7. Hard constraints

- **Attention backend must be `triton`.** The server refuses to start with
  anything else (e.g. `--tts_engine.engine.attention_backend fa3`).
- `task_type` must be `None` or `"Base"`.
- `voice` must be `"default"`, `""`, or a registered uploaded voice.
  **`"alloy"` and other OpenAI voice names are rejected.**
- `seed`, `stage_params`, `stage_sampling` are explicitly unsupported (HTTP 400).
- Triton sampling and fused tower layers are **Hopper-only (sm_90)**; they fall
  back silently on other GPUs and the startup log names the path taken.

---

## 8. Endpoint map

`/v1/chat/completions` exists but **cannot serve Drama 3** — it lowers `messages`
to a list of dicts, while the model requires `input` to be a string. Use
`/v1/audio/speech`.

| Endpoint | Purpose |
|---|---|
| `POST /v1/audio/speech` | **Text → speech (use this)** |
| `POST /v1/audio/speech/batch` | Batch variant |
| `GET /v1/audio/speech/{request_id}` | Retrieve async result |
| `GET/POST /v1/audio/voices` | List / register uploaded voices |
| `POST /v1/audio/transcriptions` | ASR (other models) |
| `GET /health` | `200` healthy / `503` starting |

---

## 9. OICM / container notes

- Runs as non-root uid `10000`, gid `0`, **read-only rootfs**; `/tmp` writable.
- `HOME` and all caches are re-pointed at `/tmp` by `startup.sh`, because the
  base image's `FLASHINFER_WORKSPACE_BASE` would otherwise target a read-only
  path.
- Offline: if `codec.pth` is present in the model directory, `startup.sh` passes
  `--vocoder.factory.codec_path=<model dir>` so no Hugging Face lookup happens.
- Exposes port `8080`; `/health` satisfies both the OICM Service and the k8s
  probe.

---

## 10. Known harmless log noise

| Message | Meaning |
|---|---|
| `Failed to import torchada` | Optional MUSA/AMD shim. Irrelevant on NVIDIA. |
| `bsgemm-moe quantization is not fully optimized yet` | Informational. |
| `using attn output gate!` | Normal for this architecture. |
| `Config file not found ... triton_3_8_0 ... falling back to 3.6.0` | Base image ships Triton 3.8.0 but MoE kernel configs only exist for 3.6.0. Works, slightly untuned. Inherited from `lmsysorg/sglang`, not our layers. |
| `Flax classes are deprecated` | Diffusers import notice. |

---

## References

- Cookbook: `docs/cookbook/fishaudio_drama3.md` in `AITDEV00/sglang-omni` @ `v0.2.0`
- Pinned commit: `e8c635ecb9c68bcf2fbc32ceb3c40f85f3590657`
- [Upstream cookbook (published)](https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_drama3.html)
- [Fish Speech technical report](https://github.com/fishaudio/fish-speech/tree/main/docs/assets/Fish_Audio_Drama_3_Technical_Report.pdf)
