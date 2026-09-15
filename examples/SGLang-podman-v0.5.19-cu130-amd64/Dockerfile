# =============================================================================
# OICM SGLang serving image (production) -- CUDA 13.0 variant.
# Built once, pushed to your internal Harbor, reused across models. The model
# is injected at runtime by OICM's init container (object storage -> PVC), so
# this image is purely the serving runtime.
#
# WHY cu130 INSTEAD OF cu129: the upstream CUDA forward-compat contract is that
# the user-mode libcuda shipped in the image must be >= the kernel driver on the
# node. A cu129 image carries an R575-era libcuda; the cu130 image carries an
# R580-era libcuda (verified: /usr/local/cuda/compat/libcuda.so.580.126.20),
# which also satisfies newer R580+ drivers NATIVELY. Same SGLang 0.5.19 code in
# both; only the bundled CUDA toolkit + driver libs differ.
# =============================================================================

# cu130 build (CUDA 13.0.3). The runtime decides CUDA forward-compat per-node
# (see startup.sh): nodes on R580+/CUDA 13 run this image natively, while any
# older nodes (e.g. R575/CUDA 12.9 or R570/CUDA 12.8) fall back to the
# forward-compat libs the base ships at /usr/local/cuda/compat.
FROM docker.io/lmsysorg/sglang:v0.5.19-cu130

# OICM's Service and k8s /health probe target 8080 (confirmed in diagnostic).
EXPOSE 8080

# Build steps need root; we drop to the non-root runtime user further down.
USER root

# git/gcc/curl for any source builds and health tooling.
RUN set -ex && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git gcc curl && \
    rm -rf /var/lib/apt/lists/*

# NOTE: CUDA forward-compatibility libs are NOT installed here -- the cu130 base
# image ALREADY ships them at /usr/local/cuda/compat/ (an R580-era libcuda
# 580.126.20 + ptxjitcompiler/nvvm; also symlinked at /usr/local/cuda-13.0/compat
# and /usr/local/cuda-13/compat). They are used ONLY on nodes whose kernel driver
# is older than the image's CUDA; startup.sh probes the host driver at runtime
# and prepends this dir to LD_LIBRARY_PATH only when needed. On nodes already new
# enough (R580+) compat is intentionally NOT loaded, because an older user-mode
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
# and execs SGLang.
ENTRYPOINT ["/app/startup.sh"]
