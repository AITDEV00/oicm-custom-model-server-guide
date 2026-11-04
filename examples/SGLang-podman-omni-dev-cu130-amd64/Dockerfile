# =============================================================================
# OICM SGLang-Omni serving image -- CUDA 13.0.
#
# SGLang-Omni is the *pipeline* server (preprocessing -> engine -> vocoder) that
# serves omni-modality models: TTS (Fish Audio Drama 3 / S2-Pro), ASR, and
# audio/vision pipelines. It is a SEPARATE product from SGLang proper and speaks
# a different CLI, so the LLM/diffusion harness does not apply:
#
#   SGLang (this repo's other examples)   sglang serve | sglang.launch_server
#   SGLang-Omni (this image)              sgl-omni serve
#
# -----------------------------------------------------------------------------
# WHY THIS BASE -- follow upstream's own docker/Dockerfile, not the dev image
# -----------------------------------------------------------------------------
# Upstream ships two very different things under a confusingly similar name:
#
#   hongccc/sglang-omni:dev   a DEVELOPER convenience image: sglang 0.5.23,
#                             flashinfer 0.7.2, an AUTO_CLONE entrypoint that
#                             git-clones sglang-omni at container start, zsh.
#   lmsysorg/sglang@<digest>  the PRODUCTION base upstream's docker/Dockerfile
#                             builds FROM (its SGLANG_IMAGE arg): sglang 0.5.22,
#                             flashinfer 0.7.1, sglang-kernel 0.4.9.
#
# Starting from the dev image broke this image three separate ways, all observed
# live on the H200 cluster:
#
#  1. sglang 0.5.23 is NOT source-compatible with sglang-omni v0.2.0. Upstream
#     pins sglang==0.5.22 and its ModelRunner override keeps the old signature:
#         sglang 0.5.23: init_kv_cache_configurator(self, draft_parallel_layout=None)
#         sglang-omni  : init_kv_cache_configurator(self)
#     alloc_memory_pool calls the override WITH the argument, so the tts_engine
#     stage dies during startup:
#         TypeError: SGLModelRunner.init_kv_cache_configurator() takes 1
#         positional argument but 2 were given
#     (pod qt66m, ns adeo, 5 restarts; the 600 s "preprocessing made no startup
#     progress" watchdog message is the same death seen from another stage).
#
#  2. The dev image exports FLASHINFER_WORKSPACE_BASE=/root. Correct for its own
#     root user, but OICM runs us as uid 10000 on a READ-ONLY rootfs, so
#     flashinfer's JIT loader dies with
#         PermissionError: [Errno 13] Permission denied: '/root/.cache'
#     This base does NOT set it, so it falls back to $HOME -- which startup.sh
#     points at /tmp.
#
#  3. The dev image's ENTRYPOINT runs `git clone && pip install -e` at container
#     start unless SGLANG_OMNI_AUTO_CLONE=0. On an air-gapped cluster that is a
#     guaranteed CrashLoopBackOff.
#
# Everything sglang-omni v0.2.0 pins for the *serving stack* is ALREADY
# satisfied by this base:
#   sglang 0.5.22 | flashinfer-python 0.7.1 | sglang-kernel 0.4.9
#   torch 2.14.1+cu130 | transformers 5.19.0 | protobuf 6.33.6
# But the *project's own* dependency list is NOT: hydra-core, omegaconf,
# torchdiffeq, x-transformers, silero-vad, pydub, decord, sox, ... are declared
# in sglang-omni's [project.dependencies] and have nothing to do with sglang.
#
# Installing only the `fishaudio` extra (descript-audiotools + descript-audio-codec)
# and then the project with --no-deps was the FOURTH bug in this image:
#     ModuleNotFoundError: No module named 'hydra'
#   sglang_omni/models/fishaudio_s2_pro/stages.py:179 load_codec
#     from hydra.utils import instantiate
# descript-audiotools does not declare hydra-core, so nothing pulled it in.
#
# The recipe below is upstream's docker/Dockerfile: install the project's
# REQUIREMENTS (`-r pyproject.toml`) plus the extra in one resolved pass, with
# the project's own protobuf override. That gives hydra-core/omegaconf (used by
# DAC's instantiate) and the rest of the declared tree.
#
# A `--dry-run` against this base shows 47 packages added and 5 version bumps
# (cache-dit, omegaconf, pyyaml, tomlkit, typeguard) -- and sglang, torch,
# torchvision, torchaudio, transformers, flashinfer and protobuf all UNTOUCHED.
# Do NOT drop -r pyproject.toml to "keep it small": that is exactly the bug.
# =============================================================================

# Upstream's pinned production base (lmsysorg/sglang, NOT hongccc/sglang-omni).
# This is the SGLANG_IMAGE default from sgl-project/sglang-omni docker/Dockerfile.
FROM docker.io/lmsysorg/sglang@sha256:5fc09edb6a0bb1d5b6992b543723c66d08f86d7a507e951e87e0d05db4ac766e

# OICM's Service and k8s /health probe target 8080; sgl-omni serves
# GET /health (200 healthy / 503 starting), which satisfies both probes.
EXPOSE 8080

# Build steps need root; we drop to the non-root runtime user further down.
USER root

# -----------------------------------------------------------------------------
# LAYER ORDER -- built for fast iteration. Read before reordering anything.
# -----------------------------------------------------------------------------
# An instruction edit re-runs THAT step and everything below it. The 3.5 GB
# dependency install is the expensive layer, so it must sit ABOVE everything
# that changes often. Rarest-change-first:
#
#   1. apt (git/sox/curl)          changes ~never
#   2. fetch pyproject.toml        changes only on SGLANG_OMNI_VERSION bump
#   3. deps install  ~3.5 GB       changes only on SGLANG_OMNI_VERSION bump
#   4. install project source      changes on version bump
#   5. import + pin assertions     changes when you add a guard
#   6. runtime user                changes ~never
#   7. COPY startup.sh / normalizer  <-- EDITED CONSTANTLY, must be LAST
#   8. LABELs
#
# Editing startup.sh therefore rebuilds layers 7-8 only: a few seconds with the
# 3.5 GB deps layer cached. Do NOT merge 2-4 into one RUN "to keep it tidy" --
# that puts the pyproject fetch and the project source below the big install and
# makes every source tweak re-run 3.5 GB of wheels.
#
# Step 2 fetches pyproject.toml straight from the pinned tag instead of cloning
# first, so the deps layer depends ONLY on SGLANG_OMNI_VERSION -- not on the
# clone, and not on build context.
# -----------------------------------------------------------------------------

# git/curl to fetch the pinned source+metadata at build time; sox is the audio
# CLI upstream's own Dockerfile installs (vocoder / reference-audio handling).
RUN set -ex && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git curl ca-certificates sox && \
    rm -rf /var/lib/apt/lists/*

ARG SGLANG_OMNI_VERSION=v0.2.0
ARG SGLANG_OMNI_COMMIT=e8c635ecb9c68bcf2fbc32ceb3c40f85f3590657

# Project metadata only. Kept in its own layer so the dep install below is
# invalidated by a VERSION bump and nothing else.
#
# pyproject.toml is FLATTENED to a requirements file because `uv pip install -r`
# accepts neither a pyproject.toml nor an arbitrary .toml:
#   - named *.toml  -> "not a valid PEP 751 filename: expected pylock.toml..."
#   - any other name -> "Unexpected '[', expected '-c', '-e', '-r'..." (req file)
# The helper also appends the `fishaudio` extra's two packages by group name, so
# the pins stay sourced from the tag rather than being retyped here.
COPY --chown=0:0 pyproject_to_requirements.py /tmp/pyproject_to_requirements.py

RUN set -ex && \
    mkdir -p /tmp/omni-reqs && \
    curl -fsSL -o /tmp/omni-reqs/pyproject.toml \
        "https://raw.githubusercontent.com/sgl-project/sglang-omni/${SGLANG_OMNI_VERSION}/pyproject.toml" && \
    /opt/sglang/bin/python3 -c 'import tomllib; \
print("\n".join(tomllib.load(open("/tmp/omni-reqs/pyproject.toml","rb"))["tool"]["uv"]["override-dependencies"]))' \
        > /tmp/omni-overrides.txt && \
    echo "--- overrides ---" && cat /tmp/omni-overrides.txt && \
    /opt/sglang/bin/python3 /tmp/pyproject_to_requirements.py \
        /tmp/omni-reqs/pyproject.toml /tmp/omni-reqs/requirements.txt fishaudio && \
    echo "--- requirements (incl. fishaudio extra) ---" && \
    grep -iE "descript|hydra|torchdiffeq|x-transformers" /tmp/omni-reqs/requirements.txt

# THE BIG LAYER (~3.5 GB). Upstream's own recipe, one resolved pass:
#   -r requirements.txt  the project's declared dependencies (hydra-core,
#                        omegaconf, torchdiffeq, x-transformers, silero-vad, ...)
#                        PLUS the fishaudio extra, both flattened from the tag
#   --overrides          the project's protobuf pin, because descript-audiotools
#                        declares protobuf<3.20 and would otherwise downgrade the
#                        base's 6.33.6 and break every generated protobuf stub
#   --no-build-isolation build against the base's already-installed torch
#
# A --dry-run against this base adds 47 packages and bumps 5 (cache-dit,
# omegaconf, pyyaml, tomlkit, typeguard); sglang, torch, torchvision,
# torchaudio, transformers, flashinfer and protobuf are all left UNTOUCHED.
RUN set -ex && \
    /usr/local/bin/uv pip install --python /opt/sglang/bin/python3 \
        --no-build-isolation \
        --overrides /tmp/omni-overrides.txt \
        -r /tmp/omni-reqs/requirements.txt && \
    rm -rf /tmp/omni-reqs /tmp/omni-overrides.txt /tmp/pyproject_to_requirements.py

# The project source itself: SMALL (a few MB). The checkout is deleted in the
# same layer so nothing is left behind. Installed --no-deps because every
# dependency is present from the layer above.
RUN set -ex && \
    git clone --depth 1 --branch "${SGLANG_OMNI_VERSION}" \
        https://github.com/sgl-project/sglang-omni.git /tmp/sglang-omni-src && \
    test "$(git -C /tmp/sglang-omni-src rev-parse HEAD)" = "${SGLANG_OMNI_COMMIT}" && \
    /opt/sglang/bin/pip install --no-deps --no-build-isolation /tmp/sglang-omni-src && \
    rm -rf /tmp/sglang-omni-src && \
    /opt/sglang/bin/sgl-omni --help > /dev/null

# Import checks. These cover the three failure modes actually observed at
# runtime, so a regression fails the BUILD instead of the cluster:
#   dac/audiotools  -> `flatten_dict` (fishaudio extra installed without deps)
#   hydra           -> the DAC `instantiate` path in load_codec
#   s2_pro stages   -> the module that imports hydra at vocoder construction
RUN set -ex && /opt/sglang/bin/python3 -c "\
import dac, audiotools; \
from dac.nn.layers import Snake1d; \
import hydra; \
from hydra.utils import instantiate; \
import omegaconf, torchdiffeq, x_transformers, silero_vad, pydub; \
import sglang_omni.models.fishaudio_s2_pro.stages as s2; \
import sglang_omni.models.fishaudio_drama3.stages as d3; \
print('vocoder + hydra + drama3/s2_pro stages import OK')"

# Pin assertions: a drifted base must fail the BUILD, not a cluster rollout.
# init_kv_cache_configurator is checked by signature because that exact mismatch
# (0.5.23 vs omni's 0.5.22 pin) crashed the first deployment.
RUN set -ex && /opt/sglang/bin/python3 -c "\
from importlib.metadata import version; \
import inspect; \
assert version('sglang') == '0.5.22', version('sglang'); \
assert version('flashinfer-python') == '0.7.1', version('flashinfer-python'); \
assert version('sglang-omni') == '0.2.0', version('sglang-omni'); \
import sglang.srt.model_executor.model_runner as mr; \
sig = str(inspect.signature(mr.ModelRunner.init_kv_cache_configurator)); \
assert 'draft_parallel_layout' not in sig, sig; \
import sglang_omni.models.fishaudio_drama3.config as c; \
assert c.Drama3PipelineConfig.architecture == 'FishOmniModel'; \
print('pins OK | sglang', version('sglang'), '| flashinfer', version('flashinfer-python'), '| omni', version('sglang-omni')); \
print('init_kv_cache_configurator', sig)"

# Create the runtime user with primary group 0. OICM runs us as exactly uid
# 10000, but gid-0 + group-writable dirs keeps the image safe even if a future
# tenant policy forces a different UID.
RUN set -ex && \
    useradd --uid 10000 --gid 0 runner && \
    mkdir -p /home/runner /app && \
    chown -R 10000:0 /home/runner /app && \
    chmod -R g=u /home/runner /app && \
    passwd -l root

# Numeric USER so Kubernetes runAsNonRoot can validate it as non-root.
USER 10000

# HOME is set here for completeness; startup.sh re-points it to /tmp at runtime
# because /home/runner is read-only under OICM.
ENV HOME=/home/runner

# Keep pip-installed user binaries on PATH.
ENV PATH="$HOME/.local/bin:$PATH"

# The NVIDIA runtime injects the driver.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# stdlib arg normalizer, shared with the SGLang examples.
COPY --chown=10000:0 startup.sh /app/startup.sh
COPY --chown=10000:0 arg_normalizer.py /app/arg_normalizer.py

# World-executable so it runs regardless of which UID OICM uses.
RUN chmod 0755 /app/startup.sh /app/arg_normalizer.py

# OICM does not set a custom workdir; /app matches the original convention.
WORKDIR /app

# Entrypoint resolves the model path, points caches at /tmp, enables cuda-compat
# only when the host driver needs it, normalizes EXTRA_ARGS, and execs sgl-omni.
# Debug dump is opt-in: set DEBUG_DUMP=1 for a full post-mortem dump.
ENTRYPOINT ["/app/startup.sh"]

LABEL org.opencontainers.image.title="SGLang-Omni v0.2.0 serving (OICM)"
LABEL org.opencontainers.image.version="omni-v0.2.0-cu130-amd64"
LABEL org.opencontainers.image.base.name="docker.io/lmsysorg/sglang@sha256:5fc09edb6a0bb1d5b6992b543723c66d08f86d7a507e951e87e0d05db4ac766e"
LABEL ai.sglang-omni.version="v0.2.0"
LABEL ai.sglang-omni.commit="e8c635ecb9c68bcf2fbc32ceb3c40f85f3590657"
LABEL ai.sglang.version="0.5.22"
LABEL ai.flashinfer.version="0.7.1"
