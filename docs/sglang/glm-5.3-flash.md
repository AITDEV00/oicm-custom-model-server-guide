# SGLang GLM-5.3-Flash

- **Folder:** `examples/SGLang-podman-glm-5.3-flash-cu129-amd64/`
- **Base image:** `docker.io/lmsysorg/sglang:glm-5.3-flash`
- **Build target:** `make build-sglang-podman-glm-5.3-flash`
- **Image tag:** `localhost/oicm/sglang:glm-5.3-flash-cu129-amd64`

## Variant context

This is a **named (non-versioned)** SGLang serving image built from the
`lmsysorg/sglang:glm-5.3-flash` base tag, rather than a `vX.Y.Z-cu129` release.
It is intended for serving the **GLM-5.3-Flash** model with the HiCache
(hierarchical L2 CPU-DRAM KV caching) stack.

The Dockerfile follows the same production hardening as the versioned
`SGLang-podman-v0.5.x-cu129-amd64` builds:

- Runs as non-root `runner` uid `10000` / gid `0` (numeric `USER 10000` so
  k8s `runAsNonRoot` validates).
- Exposes port `8080` for OICM's Service + `/health` probe.
- Ships `startup.sh` + `arg_normalizer.py` (version-agnostic, shared source of
  truth).
- CUDA forward-compat resolved at runtime via the base's `/usr/local/cuda/compat/`.

See [startup.sh anatomy](../reference/startup.md) and
[arg_normalizer.py](../reference/arg_normalizer.md).

## Pushing

The tag + push commands to the **Al Ain** and **Abu Dhabi** harbors are logged
in `commands.txt`.