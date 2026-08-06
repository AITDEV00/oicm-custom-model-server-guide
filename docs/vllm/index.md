# vLLM images

Production vLLM serving images, each under `examples/vLLM-podman-<ver>-cu129-amd64/`.
Every folder contains three files:

| File | Purpose |
|------|---------|
| `Dockerfile` | Production serving image — installs `vllm[audio]==<ver>`, creates `runner` (uid 10000, gid 0), `USER 10000` |
| `startup.sh` | Hardened entrypoint — volume resolution, model discovery, CUDA forward-compat probe, EXTRA_ARGS normalization |
| `arg_normalizer.py` | Pure-stdlib normalizer that turns OICM `EXTRA_ARGS` into final argv |

There is also a non-podman `vLLM/` baseline (Dockerfile + startup.sh only).

## Versions

| Version | Base image | Notes |
|---------|-----------|-------|
| [v0.22.0](v0.22.0.md) | `vllm/vllm-openai:v0.22.0-cu129` | |
| [v0.23.0](v0.23.0.md) | `vllm/vllm-openai:v0.23.0-cu129` | |
| [v0.24.0](v0.24.0.md) | `vllm/vllm-openai:v0.24.0-cu129` | NVFP4 W4A4 models |
| [v0.25.0](v0.25.0.md) | `vllm/vllm-openai:v0.25.0-cu129` | |
| [v0.25.1](v0.25.1.md) | `vllm/vllm-openai:v0.25.1-cu129` | DiffusionGemma / Gemma-4 |
| [MiniMax-M3](minimaxm3.md) | `vllm/vllm-openai:minimax-m3-x86_64-cu129` | |
| [Nightly](nightly.md) | `vllm/vllm-openai:cu129-nightly-x86_64` | |