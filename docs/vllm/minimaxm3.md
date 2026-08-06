# vLLM MiniMax-M3

- **Folder:** `examples/vLLM-podman-minimaxm3-cu129-amd64/`
- **Base image:** `docker.io/vllm/vllm-openai:minimax-m3-x86_64-cu129`
- **Build target:** `make build-vllm-podman-minimaxm3`

MXFP8 serving image for MiniMax-M3. Unlike the versioned variants, this image
has **no `VLLM_VERSION` ARG** — it relies on the dedicated `minimax-m3` base
image tag.

See [startup.sh anatomy](../reference/startup.md) and
[arg_normalizer.py](../reference/arg_normalizer.md).