# =============================================================================
# OICM SGLang serving image (production).
# Built once, pushed to your internal Harbor, reused across models. The model
# is injected at runtime by OICM's init container (object storage -> PVC), so
# this image is purely the serving runtime.
# =============================================================================

# cu129 build. The runtime decides CUDA forward-compat per-node (see startup.sh):
# H200 nodes on R580/CUDA 13.0 run this image natively, while any older nodes
# (e.g. R570/CUDA 12.8) fall back to the forward-compat libs the base ships.
FROM docker.io/lmsysorg/sglang:v0.5.14-cu129

# OICM's Service and k8s /health probe target 8080 (confirmed in diagnostic).
EXPOSE 8080

# Build steps need root; we drop to the non-root runtime user further down.
USER root

# git/gcc/curl for any source builds and health tooling.
RUN set -ex && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git gcc curl && \
    rm -rf /var/lib/apt/lists/*

# bitsandbytes for NF4 / 8-bit quantization (e.g. ideogram-4-nf4). Installed as
# root into the system site-packages so the non-root runner user can import it.
# Pinned to >=0.46.1 for the kernels SGLang's bnb loader expects.
RUN pip install --no-cache-dir "bitsandbytes>=0.46.1"

# Patch SGLang's diffusion-server NVML fallback to catch NVMLError_NoPermission
# (raised on MIG-in-container), not just NVMLError_NotSupported. Upstream
# already has the torch.cuda fallback -- it just catches too narrow an exception.
# See: multimodal_gen/runtime/platforms/cuda.py, NvmlCudaPlatform.get_device_total_memory
# The assert makes a future image where that block moved fail the build loudly
# instead of silently shipping unpatched.
RUN F="$(python3 -c 'import sglang,os;print(os.path.dirname(sglang.__file__))')/multimodal_gen/runtime/platforms/cuda.py" && \
    python3 -c "import sys; f=sys.argv[1]; s=open(f).read(); old='        try:\n            return int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)\n        except pynvml.NVMLError_NotSupported:\n            return int(torch.cuda.get_device_properties(device_id).total_memory)'; new='        try:\n            return int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)\n        except pynvml.NVMLError:  # incl. NVMLError_NoPermission on MIG-in-container\n            return int(torch.cuda.get_device_properties(device_id).total_memory)'; assert s.count(old)==1, f'anchor count {s.count(old)} != 1 - source moved; re-inspect cuda.py'; open(f,'w').write(s.replace(old,new)); print('patched:', f)" "$F"

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
# and execs SGLang.
ENTRYPOINT ["/app/startup.sh"]
