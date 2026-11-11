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
| [Diagnostics](diagnostics/diagnostics.md) | `python:3.12-slim` | Runtime environment probe (no model) |
| [cputests](cputests.md) | `python:3.11-slim` | CPU + Silero-VAD benchmark |

## Building

```bash
cd examples
make build-vllm            # docker, v0.19.0
make build-vllm-podman-0.27.1  # vllm[audio] + librosa
make build-sglang-podman-0.5.18
make build-diagnostics
make build-cputests
```

See [Build & push](reference/build.md) for the full target list and registry
push workflow.

## Supported Al Ain image delivery (v0.5.23)

For a large Al Ain runtime image, use the [canonical public-HF → hfs3/MinIO →
Harbor relay runbook](../../devops-custom-models/docs/oicm/runbooks/alain-image-hf-relay-runbook.md).
The durable Docker-archive route succeeded with SGLang v0.5.23 on 2026-11-11.
It changes the registry only, not any deployment; verify the final Harbor
config/manifest/digest rather than inferring publication from a process.

The repository-relative source link leaves this site’s `docs/` tree. For
site-safe navigation use the [canonical runbook on
GitHub](https://github.com/AITDEV00/devops-custom-models/blob/main/docs/oicm/runbooks/alain-image-hf-relay-runbook.md).