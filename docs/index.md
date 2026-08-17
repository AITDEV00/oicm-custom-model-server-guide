# OICM Custom Model Server Guide

Guide to building Custom Model Servers for the OICM platform. This repo holds
the reference **Docker images / startup.sh templates** — the *contract* the
OICM platform expects when building a server image.

!!! note "Two-repo separation"
    This is the **server contract** repo. If you are looking for the deployment
    scripts, download utilities, tests, and runbooks, that is the
    [`devops-custom-models`](https://github.com/AITDEV00/devops-custom-models)
    repo. This repo only contains image definitions under `examples/`.

## Platform contract

A custom model server must satisfy the OICM platform contract. See the
[Platform contract](contract.md) page.

## Image categories

| Category | Base | Purpose |
|----------|------|---------|
| [vLLM images](vllm/index.md) | `docker.io/vllm/vllm-openai:*` | OpenAI-compatible vLLM servers |
| [SGLang images](sglang/index.md) | `docker.io/lmsysorg/sglang:*` | SGLang servers |
| [Diagnostics](diagnostics/index.md) | `python:3.12-slim` | Runtime environment probe (no model) |
| [cputests](cputests.md) | `python:3.11-slim` | CPU + Silero-VAD benchmark |

## Building

```bash
cd examples
make build-vllm            # docker, v0.19.0
make build-vllm-podman-0.25.1
make build-sglang-podman-0.5.17
make build-diagnostics
make build-cputests
```

See [Build & push](reference/build.md) for the full target list and registry
push workflow.