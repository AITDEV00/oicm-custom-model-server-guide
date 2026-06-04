# Lightweight OICM diagnostic image.
# Mirrors the REAL serving image's security surface (uid 10000, gid 0,
# /home/runner, numeric non-root USER, group-writable dirs, port 8080) so the
# probe results reflect what the real vLLM image will face. Does NOT run vLLM.
#
# NOTE: uses python:3.12-slim to stay small. If your air-gapped Harbor does not
# mirror python:3.12-slim, change the FROM line to the vLLM base you already
# use (docker.io/vllm/vllm-openai:v0.22.0-cu129) -- everything else still works.
FROM python:3.12-slim

EXPOSE 8080
USER root

RUN set -ex && \
    apt-get update && \
    # curl  -> the only external binary the probes use (HTTPS + k8s API query)
    # ca-certificates -> required for curl to validate TLS
    # Everything else the script calls (coreutils, grep, sed, awk/mawk,
    # libc-bin/getent, bash) is in Debian's priority:required base, present
    # even in -slim. Add `iproute2` here if you want ip/ss for interactive exec.
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --uid 10000 --gid 0 runner && \
    mkdir -p /home/runner /app && \
    chown -R 10000:0 /home/runner /app && \
    chmod -R g=u /home/runner /app

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER 10000
ENV HOME=/home/runner
ENV PATH="$HOME/.local/bin:$PATH"

COPY --chown=10000:0 startup.sh /app/startup.sh
COPY --chown=10000:0 serve_report.py /app/serve_report.py
RUN chmod 0755 /app/startup.sh /app/serve_report.py

WORKDIR /app
ENTRYPOINT ["/app/startup.sh"]