# Build & push

All image build targets live in `examples/Makefile`. Each builds
`--platform=linux/amd64` and tags `oicm/<server>:<ver>-cu129(-amd64)` →
`localhost/oicm/...`.

## Targets

| Target | Image |
|--------|-------|
| `build-vllm` | vLLM docker, v0.19.0 |
| `build-vllm-podman-0.22.0` … `0.25.1` | vLLM podman versions |
| `build-vllm-podman-minimaxm3` | MiniMax-M3 |
| `build-vllm-podman-nightly` | Nightly |
| `build-sglang` | SGLang docker, v0.5.9 |
| `build-sglang-podman-0.5.14/0.5.15/0.5.16` | SGLang podman versions |
| `build-cputests` | cputests diagnostic |
| `build-diagnostics` | diagnostics image |

## Push workflow

The `commands.txt` file logs tag + push commands to:

- **Al Ain** registry — `registry.adeoaiengine.ecouncil.ae/...`
- **Abu Dhabi** Harbor — `harbor.ai.ecouncil.ae`

Some commented-out `podman save` / `rsync` transfer steps reference the remote
`adeo@10.34.104.99` host for air-gapped transfer.

!!! warning "Credentials"
    `commands.txt` contains registry login credentials. It must remain
    uncommitted / gitignored.