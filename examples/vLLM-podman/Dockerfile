# =============================================================================
# OICM vLLM serving image (production).
# Built once, pushed to your internal Harbor, reused across models. The model
# is injected at runtime by OICM's init container (object storage -> PVC), so
# this image is purely the serving runtime.
# =============================================================================

# cu129 build: there is no vLLM 0.22.0 cu128 image, and the node driver is 570
# (CUDA 12.8), so we keep cu129 and bridge the gap with CUDA forward-compat below.
FROM docker.io/vllm/vllm-openai:v0.22.0-cu129

# Pin the vLLM version so the [audio] reinstall below matches the base exactly.
ARG VLLM_VERSION=0.22.0

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

# NOTE: CUDA 12.9 forward-compatibility libs are NOT installed here -- the
# vllm:0.22.0-cu129 base image ALREADY ships them at /usr/local/cuda/compat/
# (verified: libcuda.so.575.57.08 + libnvidia-ptxjitcompiler/nvvm from R575).
# These R575 user-mode driver libs let the cu129 runtime (Triton JIT + CUDA
# graph capture) work on the node's older R570 / CUDA 12.8 kernel driver, which
# is supported because the GPU is a datacenter H200. startup.sh prepends this
# dir to LD_LIBRARY_PATH at runtime so it loads ahead of the injected 570 libs.

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

# Entrypoint script resolves model path, fixes caches, enables cuda-compat,
# normalizes EXTRA_ARGS via /app/arg_normalizer.py, and execs vLLM.
ENTRYPOINT ["/app/startup.sh"]
