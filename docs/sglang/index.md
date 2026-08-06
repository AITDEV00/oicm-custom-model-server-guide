# SGLang images

SGLang serving images, each under `examples/SGLang-podman-<ver>-cu129-amd64/`.
Each folder contains `Dockerfile`, `startup.sh`, and `arg_normalizer.py`.
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