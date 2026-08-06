# OICM Platform Contract

The **hard requirements** every Custom Model Server image must satisfy:

- Server should run on port **`8080`**
- Should be **non-root** user with **`UID 10000`**
- Model should be consumed from the path `"$PVC_PATH"` — follow the provided
  `startup.sh`
- Model ID for the OpenAI-compatible endpoint is taken from the **`MODEL_ID`**
  environment variable
- Should have a relevant health-check endpoint (system defaults to `/health`)

## Example Dockerfile

```dockerfile
FROM vllm/vllm-openai:v0.19.0

EXPOSE 8080

USER root

RUN <<eot
    set -ex
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git gcc curl
    rm -rf /var/lib/apt/lists/*

    # Install audio capabilities
    pip install vllm[audio]==${VLLM_VERSION} --no-cache-dir

    # Security: add user runner
    useradd --uid 10000 runner
    mkdir /home/runner && chown runner /home/runner
    mkdir -p /app && chown runner /app

    # Security: disable root login
    passwd -l root
EOT

USER runner
ENV HOME /home/runner
ENV PATH "$HOME/.local/bin:$PATH"
```

## Env vars used by startup.sh

| Variable | Required | Purpose |
|----------|----------|---------|
| `MODEL_ID` | Yes | Served model name for the OpenAI-compatible endpoint |
| `PVC_PATH` | Yes | Model weights volume path |
| `USE_DATA_VOLUME` | No | When `True`, load directly from `$PVC_PATH`; otherwise expect `$PVC_PATH/app/download/base_model` |
| `EXTRA_ARGS` | No | OICM "Model Server Arguments" field — normalized by `arg_normalizer.py` |