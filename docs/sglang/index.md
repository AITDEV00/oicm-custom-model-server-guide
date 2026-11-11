# SGLang images

SGLang serving images, each listed version under its own
`examples/SGLang-podman-<ver>-<cuda>-amd64/` directory. Each listed version
folder contains `Dockerfile`, `startup.sh`, and `arg_normalizer.py`.
There is also a non-podman `SGLang/` baseline (Dockerfile + startup.sh only).

The SGLang `startup.sh` variants are ~600 lines and route **LLM/embedding vs
diffusion** based on the model manifest basename, then launch
`python3 -m sglang.launch_server` on port 8080. The `arg_normalizer.py`
variants enumerate SGLang's JSON flags (e.g. `--json-model-override-args`) and
note that SGLang speculative decoding uses scalar flags, not JSON.

## Versions

| Version | Base image | Notes |
|---------|-----------|-------|
| [v0.5.14](v0.5.14.md) | `docker.io/lmsysorg/sglang:v0.5.14-cu129` | |
| [v0.5.15](v0.5.15.md) | `docker.io/lmsysorg/sglang:v0.5.15.post1-cu129` | GLM-5.2 HiCache |
| [v0.5.16](v0.5.16.md) | `docker.io/lmsysorg/sglang:v0.5.16-cu129` | DeepSeek-V4 + DSpark |
| [v0.5.17](v0.5.17.md) | `docker.io/lmsysorg/sglang:v0.5.17-cu129` | |
| [v0.5.18](v0.5.18.md) | `docker.io/lmsysorg/sglang:v0.5.18-cu129` | Qwen3.5-122B & Qwen3.6-35B HiCache (PR #33639) |
| v0.5.23 | `docker.io/lmsysorg/sglang:v0.5.23-cu130` | [Supported Al Ain HF relay](#supported-al-ain-hf-relay-v0523), verified 2026-11-11 |
| [GLM-5.3-Flash](glm-5.3-flash.md) | `docker.io/lmsysorg/sglang:glm-5.3-flash` | GLM-5.3-Flash serving (named variant) |

## Supported Al Ain HF relay (v0.5.23)

Use the [canonical durable-archive → public HF → hfs3/MinIO → containerd →
Harbor runbook](../../../devops-custom-models/docs/oicm/runbooks/alain-image-hf-relay-runbook.md)
for a cold transfer with authorized public-HF transit. The [v0.5.23
example](../../examples/SGLang-podman-v0.5.23-cu130-amd64/HF_RELAY.md) belongs
with this supported route. Completion requires actual image hash and registry
manifest/digest readback; it is not a model deployment.

These repository-relative source links leave the MkDocs `docs/` tree and are
not additional site pages. For site navigation and site-safe cross-repo
linking, use the [canonical runbook on
GitHub](https://github.com/AITDEV00/devops-custom-models/blob/main/docs/oicm/runbooks/alain-image-hf-relay-runbook.md).