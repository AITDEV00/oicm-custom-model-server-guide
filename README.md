# OICM Custom Model Server Guide 🐳

Guide to building Custom Model Server for the OICM platform.

## Hard Requirements

- Server should run on port `8080`
- Should be non-root user with `UID 10000`
- Model should be consumed from the path: `/opt/oicm/model`
- Model ID for the OpenAI compatible endpoint should be taken as `MODEL_ID` environment variable.


## Example: 

### Dockerfile

```dockerfile
FROM vllm/vllm-openai:v0.19.0

EXPOSE 8080-9000

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
python3 -m vllm.entrypoints.openai.api_server \
  --host=0.0.0.0 \
  --port=8080 \ # <--------------- Fixed Port for the server
  --model=/opt/oicm/model \ # <--- Fixed Path for Model
  --served-model-name=$MODEL_ID
```