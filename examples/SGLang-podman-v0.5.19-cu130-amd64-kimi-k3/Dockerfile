# =============================================================================
# OICM SGLang serving image -- Kimi-K3 kv-free fix variant.
#
# Base release commit : 0bcd822377 (v0.5.19; annotated tag object 59f20bffdd)
# PR #38982 snapshot  : 9bf4ab4928
# Backport commit     : 4a57e0e164
#
# Delta: ONLY sgl-project/sglang PR #38982 ([mem_cache] Coalesce touching
#        kv-row segments before free), backported onto the v0.5.19 tag and
#        frozen as kimi-k3-v0.5.19.patch.
#
# What it fixes: UnifiedRadixCache.cache_finished_req frees a request aborted
# mid-prefill as two ranges that touch at effective_len. Handed to the
# allocator separately, the second starts mid-page and the shared boundary
# page is mishandled -- on PR-branch main _page_disjoint asserts (every
# scheduler rank dies); on v0.5.19 the head-trim silently leaks the boundary
# page's slots. Either way the request abort breaks the row. Observed on
# Kimi-K3 (TP8, DSPARK, page_size 64) three times in production in two days.
#
# What is deliberately NOT included:
#   - GLM-5.x DSA/DCP changes (PR #31821) -- see the ...-glm53-dcp image
#   - GLM-5.3-Flash PR #36507
#   - any other post-v0.5.19 main commits
#
# The patch overlays the editable /sgl-workspace/sglang source tree; the
# dependency baseline (PyTorch, FlashInfer 0.6.18, sglang-kernel 0.4.6.post1)
# is untouched.
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
# sha256: 57aa48c66615526b59b15e8f5b16756f816e0f93daa33eaec8ac4d21b23555d0
ARG KIMI_K3_PATCH_SHA256="57aa48c66615526b59b15e8f5b16756f816e0f93daa33eaec8ac4d21b23555d0"

# Dry-run first so a drifted base image fails the build instead of producing a
# silently half-patched tree.
COPY kimi-k3-v0.5.19.patch /tmp/kimi-k3-v0.5.19.patch
RUN set -ex && \
    echo "${KIMI_K3_PATCH_SHA256}  /tmp/kimi-k3-v0.5.19.patch" | sha256sum -c - && \
    cd /sgl-workspace/sglang && \
    patch --dry-run --batch --forward -p1 < /tmp/kimi-k3-v0.5.19.patch && \
    patch --batch --forward -p1 < /tmp/kimi-k3-v0.5.19.patch && \
    rm /tmp/kimi-k3-v0.5.19.patch && \
    python3 -m compileall -q python/sglang

# Import sanity check: catches missing-symbol / version-skew breakage that
# compileall (syntax only) cannot. (Script file, not a heredoc: podman's
# builder misparses heredoc bodies that start with "import".)
COPY import_check.py /tmp/import_check.py
RUN python3 /tmp/import_check.py && rm /tmp/import_check.py

# Run the PR's unit test (adapted to v0.5.19 allocator semantics) at build time.
RUN cd /sgl-workspace/sglang && \
    PYTHONPATH=/sgl-workspace/sglang/test python3 -m pytest \
      test/registered/unit/mem_cache/test_free_kv_row_segments_touching.py \
      -q -p no:cacheprovider

# Runtime writes (SGLang/FlashInfer JIT caches, HF, torch) go to /tmp ONLY:
# OICM's deploy env injects HOME/XDG_CACHE_HOME/HF_HOME/TRITON_CACHE_DIR/
# TORCH_HOME/FLASHINFER_WORKSPACE_BASE/TMPDIR -> /tmp (node-local writable
# disk) and startup.sh defers to those. The rootfs is read-only, so no cache
# dirs are baked into the image. For standalone ad-hoc runs pass:
#   podman run -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache ...

LABEL org.opencontainers.image.title="SGLang v0.5.19 + Kimi-K3 kv-free fix"
LABEL org.opencontainers.image.version="0.5.19-cu130-amd64-kimi-k3-4a57e0e164"
LABEL org.opencontainers.image.base.name="docker.io/lmsysorg/sglang:v0.5.19-cu130@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9"
LABEL ai.sglang.base="v0.5.19 (0bcd822377)"
LABEL ai.sglang.patch="PR-38982 (9bf4ab4928)"
LABEL ai.sglang.patch-sha256="57aa48c66615526b59b15e8f5b16756f816e0f93daa33eaec8ac4d21b23555d0"
LABEL ai.sglang.backport-commit="4a57e0e164"

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
