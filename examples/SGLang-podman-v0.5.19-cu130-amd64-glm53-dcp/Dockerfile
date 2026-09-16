# =============================================================================
# OICM SGLang serving image -- GLM-5.3 DSA/DCP variant.
#
# Base release commit : 0bcd822377 (v0.5.19; annotated tag object 59f20bffdd)
# PR #31821 snapshot  : 58e13e93e3
# Backport commit     : f1eb59f54f
#
# Delta: ONLY sgl-project/sglang PR #31821 (GLM-5.x DSA DCP), backported onto
#        the v0.5.19 tag and frozen as glm53-dcp-v0.5.19.patch.
#
# What is deliberately NOT included:
#   - Kimi-K3 fixes (PR #38982) -- see the ...-kimi-k3 image
#   - GLM-5.3-Flash PR #36507
#   - any other post-v0.5.19 main commits
#
# The patch overlays the editable /sgl-workspace/sglang source tree; the
# dependency baseline (PyTorch, FlashInfer 0.6.18, sglang-kernel 0.4.6.post1)
# is untouched and already satisfies PR #31821's FlashInfer >= 0.6.15
# requirement for EAGLE+DCP.
#
# The patch includes runtime-JIT C++ (fixup_zero_kv.cuh); compileall + imports
# here validate Python only -- real CUDA/JIT compilation happens at first use
# on the GPU node.
# =============================================================================

# Digest-pinned so the base cannot change under an identical tag.
# digest sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9
FROM docker.io/lmsysorg/sglang:v0.5.19-cu130@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9

USER root

RUN set -ex && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends patch && \
    rm -rf /var/lib/apt/lists/*

# Fingerprint of the frozen patch bytes. If you regenerate the patch, update
# this ARG -- the build fails on mismatch on purpose (reproducibility).
# sha256: 99652263347a6ed7b09674ad450753fc8b7270c9db56da5a0675ca82a53fdab1
ARG GLM53_DCP_PATCH_SHA256="99652263347a6ed7b09674ad450753fc8b7270c9db56da5a0675ca82a53fdab1"

# Dry-run first so a drifted base image fails the build instead of producing a
# silently half-patched tree.
COPY glm53-dcp-v0.5.19.patch /tmp/glm53-dcp-v0.5.19.patch
RUN set -ex && \
    echo "${GLM53_DCP_PATCH_SHA256}  /tmp/glm53-dcp-v0.5.19.patch" | sha256sum -c - && \
    cd /sgl-workspace/sglang && \
    patch --dry-run --batch --forward -p1 < /tmp/glm53-dcp-v0.5.19.patch && \
    patch --batch --forward -p1 < /tmp/glm53-dcp-v0.5.19.patch && \
    rm /tmp/glm53-dcp-v0.5.19.patch && \
    python3 -m compileall -q python/sglang

# Import sanity check: catches missing-symbol / version-skew breakage that
# compileall (syntax only) cannot. (Script file, not a heredoc: podman's
# builder misparses heredoc bodies that start with "import".)
COPY import_check.py /tmp/import_check.py
RUN python3 /tmp/import_check.py && rm /tmp/import_check.py

# Runtime writes (SGLang/FlashInfer JIT caches, HF, torch) go to /tmp ONLY:
# OICM's deploy env injects HOME/XDG_CACHE_HOME/HF_HOME/TRITON_CACHE_DIR/
# TORCH_HOME/FLASHINFER_WORKSPACE_BASE/TMPDIR -> /tmp (node-local writable
# disk) and startup.sh defers to those. The rootfs is read-only, so no cache
# dirs are baked into the image. For standalone ad-hoc runs pass:
#   podman run -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache ...

LABEL org.opencontainers.image.title="SGLang v0.5.19 + GLM-5.x DSA DCP"
LABEL org.opencontainers.image.version="0.5.19-cu130-glm53-dcp-f1eb59f54f"
LABEL org.opencontainers.image.base.name="docker.io/lmsysorg/sglang:v0.5.19-cu130@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9"
LABEL ai.sglang.base="v0.5.19 (0bcd822377)"
LABEL ai.sglang.patch="PR-31821 (58e13e93e3)"
LABEL ai.sglang.patch-sha256="99652263347a6ed7b09674ad450753fc8b7270c9db56da5a0675ca82a53fdab1"
LABEL ai.sglang.backport-commit="f1eb59f54f"

USER 10000

# OICM runtime: resolve model path, point caches at /tmp, normalize
# EXTRA_ARGS via arg_normalizer.py, then exec SGLang on 8080. Required —
# the deployment sets no command/args, so the image must launch itself
# (with the base nvidia entrypoint + empty CMD the container just execs
# bash and exits 0 -> CrashLoopBackOff).
COPY --chown=10000:0 startup.sh /app/startup.sh
COPY --chown=10000:0 arg_normalizer.py /app/arg_normalizer.py
RUN chmod 0755 /app/startup.sh /app/arg_normalizer.py
WORKDIR /app
ENTRYPOINT ["/app/startup.sh"]

# OICM's Service and k8s probes target 8080 (startup.sh passes --port 8080).
EXPOSE 8080
