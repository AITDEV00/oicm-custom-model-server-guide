# OICM Custom Model Server Guide 🐳

Guide to building Custom Model Server for the OICM platform.

## Hard Requirements

- Server should run on port `8080`
- Should be non-root user with `UID 10000`
- Model should be consumed from the path: `"$MODEL_DOWNLOAD_FOLDER"`. Should  follow the provided script: `startup.sh`
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

```bash
#!/bin/bash
if [[ "${INIT_CONTAINER_USE_PVC}" == "True" ]]; then
    BASE_PATH=${PVC_PATH:-"/pvc-home"}
    echo "INIT_CONTAINER_USE_PVC is set. Using PVC path: ${BASE_PATH}"
elif [[ "$USE_DATA_VOLUME" == "True" ]]; then
    BASE_PATH=${PVC_PATH:-"/data-volume"}
    echo "USE_DATA_VOLUME is set. Using DATA VOLUME path: ${BASE_PATH}"
else
    BASE_PATH="/tmp/oicm"
    mkdir -p "${BASE_PATH}" || { echo "Failed to create directory: ${BASE_PATH}"; exit 1; }
    echo "INIT_CONTAINER_USE_PVC is not set. Using temporary path: ${BASE_PATH}"
fi
export BASE_PATH

export BASE_DOWNLOAD_FOLDER="${BASE_PATH}/app/download"

if [[ "$USE_DATA_VOLUME" == "True" ]]; then
    MODEL_DOWNLOAD_FOLDER="${BASE_PATH}"
else
    DEFAULT_MODEL_DOWNLOAD_FOLDER="${BASE_DOWNLOAD_FOLDER}/base_model"
    MODEL_DOWNLOAD_FOLDER="${MODEL_DOWNLOAD_FOLDER:-${DEFAULT_MODEL_DOWNLOAD_FOLDER}}"
fi


python3 -m vllm.entrypoints.openai.api_server \
  --host=0.0.0.0 \
  --port=8080 \ # <--------------- Fixed Port for the server
  --model="$MODEL_DOWNLOAD_FOLDER" \ # <--- Fixed Path for Model
  --served-model-name=$MODEL_ID
```