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

## Getting Help

```bash 
make help 

# OR 

make
```