# vLLM Nightly

- **Folder:** `examples/vLLM-podman-nightly-cu129-amd64/`
- **Base image:** `docker.io/vllm/vllm-openai:cu129-nightly-x86_64`
- **Build target:** `make build-vllm-podman-nightly`

Bleeding-edge nightly serving image — used to validate new vLLM features ahead
of a pinned release (e.g. GLM-5.2 MTP speculative decoding on `nightly`).
No `VLLM_VERSION` ARG.

See [startup.sh anatomy](../reference/startup.md) and
[arg_normalizer.py](../reference/arg_normalizer.md).