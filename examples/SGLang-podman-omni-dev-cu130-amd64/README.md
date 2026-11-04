# SGLang-Omni v0.2.0 — OICM serving image (CUDA 13.0)

Serving image for **SGLang-Omni**, the omni-modality *pipeline* server
(`sgl-omni serve`). It is a separate product from SGLang proper and speaks a
different CLI, so the LLM/diffusion harness in the other `SGLang-podman-*`
examples does **not** apply here.

| | SGLang (other examples) | SGLang-Omni (this image) |
|---|---|---|
| Command | `sglang serve` / `sglang.launch_server` | `sgl-omni serve` |
| Config | flat CLI flags | dotted per-stage flags (`--tts_engine.…`) |
| Health | `/health` | `/health` (200 healthy, 503 starting) |

First model validated: **`fishaudio/drama-3`** (35B MoE TTS, `FishOmniModel`).

## Build

```bash
make build-sglang-podman-omni-dev
# -> oicm/sglang-omni:0.2.0-cu130-amd64  (localhost/oicm/sglang-omni:0.2.0-cu130-amd64)
```

## Design notes (why the Dockerfile looks the way it does)

**Base is digest-pinned, source is tag-pinned.** Upstream publishes only the
`dev` tag on Docker Hub — there is no `v0.2.0` image. `dev` tracks `main`, so
the `FROM` is pinned by digest; SGLang-Omni itself is pinned to the `v0.2.0`
release tag **and** the build asserts the resulting commit is
`e8c635ecb9`. Both pins are needed: one alone is not reproducible.

**`pip install --no-deps`, deliberately.** The base already carries every
`sglang-omni==0.2.0` requirement (torch 2.14.1+cu130, transformers 5.19.0,
flash-attn-4, flashinfer 0.7.2, nixl-cu13, torch-memory-saver) including the
`fishaudio` extra's `descript-audio-codec` + `descript-audiotools`. A
dependency-resolving install would try to **downgrade sglang 0.5.23 → 0.5.22**
(v0.2.0's pin) and pull a multi-GB torch tree for nothing. Verified: the Drama 3
pipeline config resolves, imports, and `sgl-omni config view` succeeds on 0.5.23.

**Non-editable install, source tree deleted.** The base's own entrypoint does
`git clone && pip install -e` at container start — a guaranteed
CrashLoopBackOff on an air-gapped cluster. `SGLANG_OMNI_AUTO_CLONE=0` is baked
in, the package is installed into site-packages, and the checkout is removed in
the same layer so nothing can silently depend on it.

**Offline `codec.pth` (the one real air-gap trap).** The Drama 3 vocoder
defaults its codec to a *Hub* repo:

```
--vocoder.factory.codec_path fishaudio/s2-pro@1de9996b6be38b745688de084d87a5633f714e4e
```

That download cannot succeed on the cluster. The published `fishaudio/drama-3`
checkpoint ships the **same** `codec.pth` (identical LFS oid
`74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f`, 1.87 GB),
so `startup.sh` detects `codec.pth` in the resolved model dir and points the
vocoder at it. No Hub lookup, no patch to upstream.

## OICM configuration

| Setting | Value |
|---|---|
| Image | `registry.adeoaiengine.ecouncil.ae/…/mlops-serving/sglang-omni:0.2.0-cu130-amd64` |
| Port | **8080** (OICM Service + k8s probes; `sgl-omni`'s own default is 8000) |
| Health probe | `/health` |
| Runs as | uid 10000, read-only rootfs, writes only to `/tmp` |
| `/dev/shm` | 5 GiB is the floor; sgl-omni relays frames through POSIX shm |
| GPU | 1× H200 for `fishaudio/drama-3` (engine 0.70 + vocoder 0.10 share) |

### Model Server Arguments (`EXTRA_ARGS`)

sgl-omni addresses its pipeline config with **dotted leaf flags**, one value per
flag. The stage name is the head — the `stages.` prefix is rejected.

```
--tts_engine.factory.max_running_requests 64
--tts_engine.gpu_memory_fraction 0.85
--preprocessing.factory.max_reference_audio_seconds 90
--preprocessing.total_reserve_bytes 9GiB
--vocoder.gpu 1
```

Raising `max_reference_audio_seconds` **must** be paired with a larger
`preprocessing.total_reserve_bytes` — the reserve is sized around the 60 s
default and an oversized reference OOMs the preprocessing stage.

Two different values for the same flag in one launch are **refused** (not
last-wins, unlike argparse).

## Smoke test

```bash
podman run --rm -d --name omni \
  --device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/wsl/drivers/nv_dispi.inf_amd64_b20cc8aeaed64fc2 \
  --shm-size=32g --network host \
  -v <model-dir>:/pvc-home/model:ro \
  -e INIT_CONTAINER_USE_PVC=True -e PVC_PATH=/pvc-home \
  localhost/oicm/sglang-omni:0.2.0-cu130-amd64

curl -s localhost:8080/health
curl -s localhost:8080/v1/models
```

Plain TTS:

```bash
curl -X POST localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello, how are you?"}' --output out.wav
```

Voice cloning adds `references` (a clip ≤ 60 s plus its transcript):

```bash
curl -X POST localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Get the trust fund to the bank early.",
       "references":[{"audio_path":"ref.wav","text":"We asked over twenty different people."}]}' \
  --output clone.wav
```

> On the H200 cluster use `--device nvidia.com/gpu=all` / `--gpus all` instead of
> the `/dev/dxg` WSL2 workaround shown above.

## Verify before pushing

```bash
IMG=localhost/oicm/sglang-omni:0.2.0-cu130-amd64

podman run --rm --entrypoint /opt/sglang/bin/sgl-omni "$IMG" --help
podman run --rm --entrypoint /opt/sglang/bin/python3 "$IMG" -c \
  "from sglang_omni.models.fishaudio_drama3.config import Drama3PipelineConfig as C; print(C.architecture)"

# pipeline resolution on a local dir (offline, no GPU)
podman run --rm --entrypoint /opt/sglang/bin/sgl-omni "$IMG" \
  config view --model-path /pvc-home/model
```
