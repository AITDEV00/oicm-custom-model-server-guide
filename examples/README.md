# Custom Model Server Examples

## Prerequisites

- `make` or `cmake`
- `docker` runtime
- internet connection


## Build Custom vLLM Model Server

```bash 
make build-vllm
```


## Build Custom SGLang Model Server

```bash 
make build-sglang
```


## Build Custom SGLang-Omni Model Server

SGLang-Omni is a **separate product** from SGLang: it serves omni-modality
pipelines (TTS / ASR / audio-vision) with `sgl-omni serve`, its own CLI, and
dotted per-stage flags. See
[`SGLang-podman-omni-dev-cu130-amd64/`](SGLang-podman-omni-dev-cu130-amd64/README.md)
for the image and its OICM configuration.

```bash
make build-sglang-podman-omni-dev
make push-sglang-podman-omni-dev
```

## Versioned SGLang examples

Each named version/variation is documented in its own directory. In particular,
[v0.5.23 CUDA 13.0](SGLang-podman-v0.5.23-cu130-amd64/README.md) separates
non-root image build files (`Dockerfile`, `startup.sh`, `arg_normalizer.py`)
from relay helpers under `scripts/` and declarative private-cluster configs.
Its raw transfer archive, upload logs and Python caches stay local and ignored.
The optional `.dockerignore` here applies only to builds that use the entire
`examples/` tree. Named version builds pass a single image's directory as their
context and retain their normal three input files; inspect each image's own
`Dockerfile` for extra archived patches or assets it intentionally includes.

```bash
# Run from this examples directory.
make build-sglang-podman-0.5.23-cu130
make -n verify-sglang-podman-0.5.23-cu130
```

Use `-n` for a dry-run; make without `-n` calls the corresponding image or
transfer workflow. Registry verification stays read-only and no model workload
is changed by the published SGLang relay documentation.

## Getting Help

```bash 
make help 

# OR 

make
```