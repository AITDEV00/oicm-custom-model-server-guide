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