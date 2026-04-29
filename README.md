# OICM Custom Model Server Guide 🐳

Guide to building Custom Model Server for the OICM platform.

## Hard Requirements

- Server should run on port `8080`
- Should be non-root user with `UID 10000`
- Model should be consumed from the path: `"$PVC_PATH"`. Should  follow the provided script: `startup.sh`
- Model ID for the OpenAI compatible endpoint should be taken as `MODEL_ID` environment variable.
- Should have relevant health check endpoint, system defaults to `/health`.


## Example: 

### Dockerfile

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

eot

USER runner
ENV HOME /home/runner
ENV PATH "$HOME/.local/bin:$PATH"

COPY --chown=runner startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

WORKDIR /app

ENTRYPOINT ["/app/startup.sh"]
```

### Startup Script

The `startup.sh` script bootstraps the model server and launches the OpenAI-compatible API server. Copy it into your image as shown in the Dockerfile example above.

**Environment Variables:**

| Variable          | Required | Description                                                                                                                          |
| ----------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `MODEL_ID`        | ✅ Yes    | Model name exposed on the OpenAI-compatible endpoint (`/v1/models`, etc.)                                                            |
| `PVC_PATH`        | ✅ Yes    | Path to the mounted volume containing model weights. Injected by the platform.                                                       |
| `USE_DATA_VOLUME` | No       | When `True`, the model is loaded directly from `$PVC_PATH`. Otherwise, the model is expected at `$PVC_PATH/app/download/base_model`. |


**Example**:

```bash
#!/bin/bash
python3 -m vllm.entrypoints.openai.api_server \
  --host=0.0.0.0 \
  --port=8080 \ # <--------------- Fixed Port for the server
  --model="$PVC_PATH" \ # <--- Fixed Path for Model
  --served-model-name=$MODEL_ID
```