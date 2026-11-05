# =============================================================================
# OICM SGLang serving image — Kimi-K3.1-Pro (DFLASH spec-decode capable).
#
# WHY THIS IS A DERIVED IMAGE AND NOT A PLAIN v0.5.23 BUILD
# --------------------------------------------------------
# Kimi-K3.1-Pro is NOT servable from any tagged SGLang release. It needs code
# that does not exist in v0.5.23, so this is not a "nice to have" optimisation
# layer — a stock release cannot load the model at all:
#
#   v0.5.23  models/kimi_k3.py  EntryClass = [KimiK3ForConditionalGeneration,
#                                              KimiK3LinearForCausalLM]
#                               -> no KimiK31ForConditionalGeneration
#   v0.5.23  'kimi_k31' in srt/configs/model_config.py   -> 0 hits
#
# The model's own config.json declares:
#   architectures = ["KimiK31ForConditionalGeneration"]
#   model_type    = "kimi_k31"
# so on v0.5.23 SGLang fails at model-class resolution before any kernel runs.
#
# Upstream support lives on the `kimi-k3.1-day-0` branch (2 commits ahead of
# main, 636 behind) and in `docker/k31.Dockerfile` on that branch. The cookbook
# states plainly: "Kimi K3.1 Pro is served from the `kimi-k3.1-day-0` branch".
#
# This image therefore starts FROM the official, published K3.1 image (which
# upstream builds with exactly that Dockerfile) and adds ONLY the OICM runtime
# harness (startup.sh + arg_normalizer.py). We do NOT rebuild the kernels:
# upstream already did it, and rebuilding FlashMLA/FlashInfer/DeepGEMM/Dynamo
# would need ~24 GB RAM, a Rust 1.92 toolchain and a matching nightly base.
#
# WHAT THE BASE IMAGE PROVIDES (asserted at build time below)
# ----------------------------------------------------------
# Verified by /opt/k31-image/verify_image.py in the base, and re-asserted here:
#   1. FlashMLA flashmla_ops   10/10 ops (stock/PyPI has 8 — missing
#                              sparse_decode_num_sm_parts and
#                              flashmla_token_pack_prepare, both K3.1-only:
#                              native 96-head sparse decode + prefill packing)
#   2. FlashInfer DynBlock     DynBlockKernelMaxNumExperts = 1024, and every
#                              installed fused_moe_trtllm_sm100 /
#                              trtllm_gen_routing module rebuilt from that
#                              header (K3.1 has 896 experts; the stock cap is
#                              lower, so TRT-LLM MoE routing would fall back)
#   3. Rust TreeCore           native radix tree extension loads
#   4. FlashInfer Prims-TS     overlay from 26bb6dc9 (alternate DSA backends)
#   5. DeepGEMM                patched build (MegaMoE decode block_m)
#   6. Dynamo 1.6.0            rebuilt with K3.1 + session-affinity edits
#
# Without 1-3 the server still STARTS AND ANSWERS CORRECTLY, just slower — the
# silent-fallback trap the upstream Dockerfile header warns about. That is why
# the checks below hard-fail the build.
#
# A NOTE ON BENCHMARK PROVENANCE
# ------------------------------
# The cookbook's published numbers were measured on a SEPARATE image built from
# commit 9ab3c28176 "plus one unmerged change", and the docs say the public
# dev-kimi-k31-pro tag "has not been checked against these commands". So: this
# image is the officially recommended one, but treat its exact performance as
# unverified until you measure it yourself.
#
# WHAT THIS IMAGE ADDS
# --------------------
#   /app/startup.sh       OICM entrypoint: resolves the model root, points all
#                         caches at /tmp (rootfs is read-only under OICM),
#                         normalizes EXTRA_ARGS, then execs launch_server on
#                         port 8080 (the port OICM's Service/health probe use).
#   /app/arg_normalizer.py  stdlib-only EXTRA_ARGS normalizer shared with the
#                         other SGLang examples.
#
# Nothing else is modified — the upstream Python tree, the compiled kernels and
# the dependency set stay byte-identical to the published image.
# =============================================================================

# Digest-pinned so the mutable `dev-kimi-k31-pro` tag cannot change the base
# under an identical build. This is the multi-arch INDEX digest; the amd64
# image manifest it points at is
#   sha256:4a098ad7414702a2271296ac148854448d10710ec3554ac93b57ef194999305a
# (config sha256:814578811527bf5d280d432e2a1e87e8f1e495accfe2b72cfa4ad5c32c0cead0,
#  pulled and verified 2026-11-05).
#
# Pinned by digest ON PURPOSE: `dev-kimi-k31-pro` is a preview tag on a
# day-0 branch and upstream will keep moving it.
FROM docker.io/lmsysorg/sglang:dev-kimi-k31-pro@sha256:67fc77d1cbe67de27ca54d874c82757a4844743fea899ea7d0df547e30c2c391

# OICM's Service and k8s /health probe target 8080 (startup.sh passes --port=8080).
# The base image exposes nothing relevant (its own server defaults to 30000).
EXPOSE 8080

# Build steps need root; we drop to the non-root runtime user further down.
USER root

# ---------------------------------------------------------------------------
# GATE 1 — the base is genuinely the K3.1 build, not a re-tagged stock image.
#
# Re-runs upstream's own verification (10/10 flashmla ops, DynBlock at 1024,
# Prims-TS overlay, patched DeepGEMM, patched Dynamo, Rust TreeCore import).
# A `docker pull` of the wrong tag, or upstream silently repointing the tag at
# a non-K3.1 build, fails here instead of at pod startup on a GPU node.
# ---------------------------------------------------------------------------
RUN set -ex && \
    test -f /opt/k31-image/verify_image.py && \
    python3 /opt/k31-image/verify_image.py

# ---------------------------------------------------------------------------
# GATE 2 — the K3.1 model class and its config hook are importable.
#
# verify_image.py checks COMPILED artifacts only; it says nothing about the
# Python model class. A stock v0.5.23 tree would pass nothing above but would
# also be caught here. These are the two things the model's config.json needs.
# ---------------------------------------------------------------------------
RUN set -ex && \
    python3 -c "import sglang; print('sglang', sglang.__version__)" && \
    python3 -c "\
from sglang.srt.models.kimi_k3 import KimiK31ForConditionalGeneration as K; \
print('KimiK31ForConditionalGeneration: OK ->', K.__name__)" && \
    python3 -c "\
import inspect, sglang.srt.configs.model_config as m; \
src = inspect.getsource(m); \
assert 'kimi_k31' in src, 'model_type kimi_k31 is not registered'; \
print('model_type kimi_k31: registered')"

# ---------------------------------------------------------------------------
# GATE 3 — every DFLASH flag the cookbook recipe uses is in the argparser.
#
# The recipe hard-depends on all of these; --enable-linear-replayssm-spec is
# mandatory (without it DFLASH keeps one intermediate KDA state per draft token
# and the KV pool collapses: 5.13M -> 0.77M tokens measured on decode), and
# --speculative-use-block-verification is what the drafter's measured numbers
# were taken with. A missing flag means a hard startup failure, so fail the
# BUILD instead and keep the broken image out of the registry.
#
# Script file, not a heredoc: podman's builder misparses heredoc bodies passed
# to RUN (the same constraint the kimi-k3 example hit with import_check.py).
#
# Left in the image at /app/k31_flags_check.py (not deleted) so it can be
# re-run against any future base, like upstream's verify_image.py.
# ---------------------------------------------------------------------------
COPY --chown=0:0 k31_flags_check.py /app/k31_flags_check.py
RUN set -ex && \
    python3 /app/k31_flags_check.py

# ---------------------------------------------------------------------------
# OICM runtime user.
#
# OICM runs the container as exactly uid 10000, and the base image has no such
# user (only ubuntu uid=1000). Gid 0 + group-writable dirs keeps the image
# usable even if a tenant policy later forces a different UID.
#
# No apt-get install is needed: git/gcc/curl/patch/python3 are already in the
# base (checked 2026-11-05). Adding packages here would risk perturbing the
# kernel stack upstream verified, so we don't.
# ---------------------------------------------------------------------------
RUN set -ex && \
    (getent group 0 >/dev/null || true) && \
    (id -u runner >/dev/null 2>&1 || useradd --uid 10000 --gid 0 runner) && \
    mkdir -p /app && \
    chown -R 10000:0 /app && \
    chmod -R g=u /app && \
    passwd -l root

# Numeric USER so Kubernetes runAsNonRoot can validate it as non-root.
USER 10000

# HOME is set for completeness; startup.sh re-points it to /tmp at runtime
# because /home/runner is read-only under OICM.
ENV HOME=/home/runner
ENV PATH="$HOME/.local/bin:$PATH"

# Let the NVIDIA runtime inject the driver.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ---------------------------------------------------------------------------
# LAYER ORTHOGONALITY — why the harness goes LAST and in ONE COPY
#
# `podman build` re-runs an instruction when its inputs change, and every
# instruction BELOW it re-runs too. So anything volatile must sit at the very
# bottom, and it must be a SINGLE COPY: one COPY is one layer, so editing
# startup.sh later touches only that one small layer.
#
# Everything above this point is pinned by the base image digest or by literal
# build-gate scripts, so it is cache-stable. Editing startup.sh/arg_normalizer.py
# therefore rebuilds ~25 KB and re-pushes ~25 KB, not 40 GB. The base's 78
# layers stay content-addressed and are never re-uploaded.
#
# `RUN chmod` is deliberately avoided: a RUN would add a second gated layer and
# re-open the cache chain. COPY --chmod sets the mode in the copy's own layer.
# ---------------------------------------------------------------------------
COPY --chown=10000:0 --chmod=0755 startup.sh /app/startup.sh
COPY --chown=10000:0 --chmod=0755 arg_normalizer.py /app/arg_normalizer.py

# OICM sets no command/args, so the image must launch itself. With the base
# nvidia entrypoint + an empty CMD the container would exec bash and exit 0 ->
# CrashLoopBackOff. Replacing ENTRYPOINT is therefore required, not cosmetic.
# It comes after the harness copies so a future entrypoint tweak re-runs nothing
# expensive either.
WORKDIR /app
ENTRYPOINT ["/app/startup.sh"]

LABEL org.opencontainers.image.title="SGLang Kimi-K3.1-Pro (DFLASH) + OICM harness"
LABEL org.opencontainers.image.version="kimi-k31-pro-cu130-amd64"
LABEL org.opencontainers.image.base.name="docker.io/lmsysorg/sglang:dev-kimi-k31-pro@sha256:67fc77d1cbe67de27ca54d874c82757a4844743fea899ea7d0df547e30c2c391"
LABEL ai.sglang.base="lmsysorg/sglang:dev-kimi-k31-pro (kimi-k3.1-day-0 branch)"
LABEL ai.sglang.k31-source-commit="afcd7d6b487a9cbe73e25941971b4cc107e6f9ae"
LABEL ai.sglang.upstream-build="docker/k31.Dockerfile (10/10 flashmla ops, DynBlock 1024, Prims-TS 26bb6dc9, DeepGEMM patched, Dynamo 1.6.0 patched)"
LABEL ai.sglang.benchmark-caveat="cookbook numbers are from a separate build at 9ab3c28176; public tag not checked against them"
