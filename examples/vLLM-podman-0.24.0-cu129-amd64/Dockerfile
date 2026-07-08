# =============================================================================
# OICM vLLM serving image (production).
# Built once, pushed to your internal Harbor, reused across models. The model
# is injected at runtime by OICM's init container (object storage -> PVC), so
# this image is purely the serving runtime.
# =============================================================================

# cu129 build. The runtime decides CUDA forward-compat per-node (see startup.sh):
# H200 nodes on R580/CUDA 13.0 run this image natively, while any older nodes
# (e.g. R570/CUDA 12.8) fall back to the forward-compat libs the base ships.
FROM docker.io/vllm/vllm-openai:v0.24.0-cu129

# Pin the vLLM version so the [audio] reinstall below matches the base exactly.
ARG VLLM_VERSION=0.24.0

# OICM's Service and k8s /health probe target 8080 (confirmed in diagnostic).
EXPOSE 8080

# Build steps need root; we drop to the non-root runtime user further down.
USER root

# git/gcc/curl for any source builds and health tooling; reinstall vllm with the
# [audio] extra (adds Whisper-style audio support) matching the base version.
RUN set -ex && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git gcc curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip install vllm[audio]==${VLLM_VERSION} --no-cache-dir

# NOTE: CUDA forward-compatibility libs are NOT installed here -- the cu129 base
# image ALREADY ships them at /usr/local/cuda/compat/ (an R575-era libcuda +
# ptxjitcompiler/nvvm). They are used ONLY on nodes whose kernel driver is older
# than the image's CUDA; startup.sh probes the host driver at runtime and prepends
# this dir to LD_LIBRARY_PATH only when needed. On nodes already new enough (e.g.
# R580/CUDA 13.0) compat is intentionally NOT loaded, because an older user-mode
# libcuda on a newer kernel module triggers CUDA error 803.

# Create the runtime user with primary group 0. OICM runs us as exactly uid
# 10000 (diagnostic confirmed), but gid-0 + group-writable dirs keeps the image
# safe even if a future tenant policy forces a different UID.
RUN set -ex && \
    useradd --uid 10000 --gid 0 runner && \
    # /home/runner and /app exist and are group-writable (g=u) for robustness;
    # note the runtime rootfs is read-only, so real writes go to /tmp anyway.
    mkdir -p /home/runner /app && \
    chown -R 10000:0 /home/runner /app && \
    chmod -R g=u /home/runner /app && \
    # Lock the root account; the container must never run privileged.
    passwd -l root

# Numeric USER so Kubernetes runAsNonRoot can validate it as non-root.
USER 10000

# HOME is set here for completeness; startup.sh re-points it to /tmp at runtime
# because /home/runner is read-only under OICM.
ENV HOME=/home/runner

# Keep pip-installed user binaries on PATH.
ENV PATH="$HOME/.local/bin:$PATH"

# Let the NVIDIA runtime inject the driver (it does; nvidia-smi was present).
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Copy the entrypoint and the arg normalizer (one source of truth, also
# imported by the diagnostic image's POST /test-args endpoint).
# Pure stdlib (shlex/json) -- no pip install, no new layer.
COPY --chown=10000:0 startup.sh /app/startup.sh
COPY --chown=10000:0 arg_normalizer.py /app/arg_normalizer.py

# World-executable so it runs regardless of which UID OICM uses.
RUN chmod 0755 /app/startup.sh /app/arg_normalizer.py

# OICM does not set a custom workdir; /app matches the original convention.
WORKDIR /app

# Entrypoint script resolves model path, fixes caches, enables cuda-compat only
# when the host driver needs it, normalizes EXTRA_ARGS via /app/arg_normalizer.py,
# and execs vLLM.
ENTRYPOINT ["/app/startup.sh"]
