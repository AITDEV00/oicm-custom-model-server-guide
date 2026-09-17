# =============================================================================
# OICM SGLang serving image -- Kimi-K3 kv-free fix variant.
#
# Base release commit : 0bcd822377 (v0.5.19; annotated tag object 59f20bffdd)
# PR #38982 snapshot  : 9bf4ab4928
# Backport tip        : f664ce5fb9 (4 additional hardening picks, see below)
#
# Delta: PR #38982 (coalesce touching kv-row segments before free) PLUS the
# HiCache/Mamba hardening set for the TP8+EP8+DCP8+HiCache-L2, no-DSPARK
# deployment:
#   - PR #34820 (13469c1, merged 2026-09-09 AFTER v0.5.19): store mamba
#     prefix-cache checkpoints at the configured SSM state dtype
#     (bf16 KDA state -> fp32 track buffer, no double-round).
#   - PR #36770 (8370c2e + af9ff67, OPEN upstream): graceful Mamba cache
#     exhaustion — skip caching a chunk instead of killing the scheduler
#     when HiCache DMA pins all candidate slots. Consciously carried
#     production hardening; drop when merged upstream.
#   - PR #38157 (8128658, OPEN upstream): read host MemAvailable once per
#     TP group before splitting the HiCache budget — prevents false
#     'Not enough host memory available' at multi-hundred-GB L2 sizes.
#
# Already present in v0.5.19 (verified via merge-base ancestry, do NOT
# re-pick): #33112 DCP+HiCache L2 (1a3bea7), #33639 Mamba branching
# (3c533ac), #34808 mamba checkpoint depth under DCP (c20acee), #35084
# DCP prefill sync removal (f44a130), #35412 decode checkpoint grid
# (eac91ac3 — mamba_track_grid is page x LCM, correct for TP8/DCP8 512),
# #36317 HiCache auxiliary load-back ownership (5263568).
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
# sha256: dc3a2fff019a24ed13231dc9e6de0dce99cdc43b7e0e0ef6e9352aa2e82d5d2b
ARG KIMI_K3_PATCH_SHA256="dc3a2fff019a24ed13231dc9e6de0dce99cdc43b7e0e0ef6e9352aa2e82d5d2b"

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

# Run the backported test set at build time: the PR #38982 unit test
# (adapted to v0.5.19 allocator semantics) plus the #34820 SSM-dtype and
# #36770 Mamba-allocation tests carried by this patch.
RUN cd /sgl-workspace/sglang && \
    PYTHONPATH=/sgl-workspace/sglang/test python3 -m pytest \
      test/registered/unit/mem_cache/test_free_kv_row_segments_touching.py \
      test/registered/unit/layers/attention/test_mamba_track_state_dtype.py \
      test/registered/unit/layers/test_mamba2_track_ssm_indices.py \
      -q -p no:cacheprovider

# Runtime writes (SGLang/FlashInfer JIT caches, HF, torch) go to /tmp ONLY:
# OICM's deploy env injects HOME/XDG_CACHE_HOME/HF_HOME/TRITON_CACHE_DIR/
# TORCH_HOME/FLASHINFER_WORKSPACE_BASE/TMPDIR -> /tmp (node-local writable
# disk) and startup.sh defers to those. The rootfs is read-only, so no cache
# dirs are baked into the image. For standalone ad-hoc runs pass:
#   podman run -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache ...

LABEL org.opencontainers.image.title="SGLang v0.5.19 + Kimi-K3 kv-free + HiCache/Mamba hardening"
LABEL org.opencontainers.image.version="0.5.19-cu130-amd64-kimi-k3-b31c387eb8"
LABEL org.opencontainers.image.base.name="docker.io/lmsysorg/sglang:v0.5.19-cu130@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9"
LABEL ai.sglang.base="v0.5.19 (0bcd822377)"
LABEL ai.sglang.patch="PR-38982 (9bf4ab4928) + PR-34820 (13469c1) + PR-36770 (8370c2e,af9ff67) + PR-38157 (8128658)"
LABEL ai.sglang.patch-sha256="dc3a2fff019a24ed13231dc9e6de0dce99cdc43b7e0e0ef6e9352aa2e82d5d2b"
LABEL ai.sglang.backport-commit="b31c387eb8"
LABEL ai.sglang.open-upstream-prs="36770,38157"

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
