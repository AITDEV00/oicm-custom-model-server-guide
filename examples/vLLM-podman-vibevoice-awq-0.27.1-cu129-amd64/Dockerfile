# =============================================================================
# OICM vLLM serving image for VibeVoice ASR (AWQ int4 split-decoder export).
#
# Base: oicm/vllm:0.27.1 (built from ../vLLM-podman-0.27.1-cu129-amd64), which
# already carries vllm[audio] + librosa on top of vllm-openai:v0.27.1-cu129.
#
# Adds on top:
#   * microsoft/VibeVoice (vibevoice + vllm_plugin packages), PATCHED for
#     vLLM 0.27.1 / transformers 5.x (see patches/vibevoice_vllm_0_27_compat.patch)
#   * diffusers (needed by the vibevoice package import chain)
#   * dist-info with the `vllm.general_plugins` entry point so vLLM loads the
#     plugin automatically in every process (engine core + API server)
#
# The model itself is injected at runtime (PVC mount) at /models/vibevoice-awq.
#
# Upstream VibeVoice commit this was built from:
#   94da20d 2026-07-24 (main)
# =============================================================================

FROM localhost/oicm/vllm:0.27.1-cu129

USER root

# ---- Layer: diffusers (only missing runtime dep of the vibevoice package) ----
RUN --mount=type=cache,target=/root/.cache/pip \
    set -ex && \
    pip install --no-deps diffusers

# ---- Layer: patched VibeVoice plugin ----------------------------------------
# The packages are copied to /opt/vibevoice and a hand-written dist-info
# registers the `vllm.general_plugins` entry point. We deliberately do NOT run
# `pip install /opt/vibevoice` because the upstream pyproject declares heavy
# deps (gradio, aiortc, ...) that are not needed for serving and would
# bloat/conflict with the pinned base env.
COPY vibevoice-src/ /opt/vibevoice/

# Write the dist-info (podman 4.x has no Dockerfile heredoc support, hence the
# python -c one-liner instead of a heredoc block).
RUN set -ex && \
    SITE=$(/usr/bin/python3 -c "import site; print(site.getsitepackages()[0])") && \
    mkdir -p "$SITE/vibevoice-1.0.0.dist-info" && \
    printf 'Metadata-Version: 2.1\nName: vibevoice\nVersion: 1.0.0\n' > "$SITE/vibevoice-1.0.0.dist-info/METADATA" && \
    printf '[vllm.general_plugins]\nvibevoice = vllm_plugin:register_vibevoice\n' > "$SITE/vibevoice-1.0.0.dist-info/entry_points.txt" && \
    touch "$SITE/vibevoice-1.0.0.dist-info/RECORD" && \
    echo "dist-info written to $SITE"

# vLLM reads PYTHONPATH via os.environ; /etc/environment is not sourced for all
# entrypoints, so set it as an ENV directive too (inherited at runtime).
ENV PYTHONPATH=/opt/vibevoice

# Sanity check at build time: plugin must import and register cleanly.
# NOTE: registration is per-process, so both checks run in ONE python3 -c.
RUN set -ex && \
    python3 -c "from vllm_plugin import register_vibevoice; register_vibevoice(); from transformers import AutoConfig; import json, tempfile, os; d=tempfile.mkdtemp(); json.dump({'model_type': 'vibevoice'}, open(os.path.join(d,'config.json'),'w')); cfg=AutoConfig.from_pretrained(d); assert type(cfg).__name__ == 'VibeVoiceConfig'; from vllm.model_executor.models.registry import ModelRegistry; assert 'VibeVoiceForASRTraining' in ModelRegistry.get_supported_archs(); print('vibevoice plugin OK: AutoConfig + vLLM ModelRegistry verified')"

# Drop back to the runtime user used by the base image.
USER 10000

# ---- Layer: VibeVoice-specific entrypoint -----------------------------------
# Split-decoder-aware model discovery (excludes decoder-awq/config.json), the
# same arg_normalizer.py contract as the base image, and AWQ-specific defaults
# (--trust-remote-code, --dtype float16). Overrides the base image's
# /app/startup.sh which does not understand the two-config layout.
COPY --chown=10000:0 startup.sh /app/startup.sh
COPY --chown=10000:0 arg_normalizer.py /app/arg_normalizer.py
RUN chmod 0755 /app/startup.sh /app/arg_normalizer.py

# Drop back to the runtime user used by the base image.
USER 10000

# Keep the base image's hardened startup.sh entrypoint (CUDA forward-compat
# probing, HF cache env, /health). It ultimately exec's `vllm serve` with the
# model path OICM mounts at /pvc-home.
WORKDIR /app
ENTRYPOINT ["/app/startup.sh"]
