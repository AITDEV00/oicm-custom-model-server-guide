# startup.sh anatomy

The `startup.sh` is the hardened entrypoint for every serving image. Key
sections (shared across vLLM and SGLang variants):

1. **Base volume resolution** — `INIT_CONTAINER_USE_PVC` / `USE_DATA_VOLUME`.
2. **Dynamic model discovery** — finds the newest `config.json` / `params.json`
   / `model_index.json` under the volume.
3. **`HOME=/tmp`** — read-only `/home/runner`.
4. **Cache dirs → `/tmp`**.
5. **Air-gap env** — `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
6. **CUDA forward-compat probe** — `IMAGE_CUDA_VERSION` (e.g. `12090`); prepends
   `/usr/local/cuda/compat` only when the host driver is older.
7. **Operator `EXTRA_ARGS`** — normalized via `arg_normalizer --argv-nul`.
8. **Tensor-parallelism detection** — MIG → `tp=1`.
9. **Launch** — `python3 -m vllm.entrypoints.openai.api_server` or
   `python3 -m sglang.launch_server` on port 8080.

The SGLang variants additionally route LLM/embedding vs diffusion based on the
manifest basename and dump full `launch_server --help` at debug.