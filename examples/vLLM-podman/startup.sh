#!/bin/bash
# =============================================================================
# OICM vLLM serving entrypoint -- hardened against the REAL OICM runtime.
# Grounded in the live diagnostic + verified follow-ups:
#
# Fixed by OICM (so handled here, in the image):
#   * Runs as uid 10000 / gid 0, HOME=/home/runner -> OICM RESPECTS the image
#     UID (not an arbitrary UID), but / and /home/runner are READ-ONLY.
#   * Root filesystem is read-only. /tmp is a large, writable, NODE-LOCAL ext4
#     disk (ephemeral: wiped on pod restart/reschedule). The model arrives on
#     the NFS PVC at /pvc-home via OICM's init container (object storage -> PVC).
#   * OICM already exports HF_HOME, TRITON_CACHE_DIR, TORCH_HOME,
#     FLASHINFER_WORKSPACE_BASE, XDG_CACHE_HOME, XDG_CONFIG_HOME, TMPDIR -> /tmp.
#     We DEFER to those and only fill the gaps OICM missed.
#   * Cluster is air-gapped (HF / PyPI unreachable) -> must run fully offline.
#   * Node driver is R570 (CUDA 12.8); the image is cu129. The base image SHIPS
#     the R575 forward-compat libs at /usr/local/cuda/compat (verified), so we
#     just put them ahead on LD_LIBRARY_PATH -- nothing to install.
#   * k8s liveness/readiness probes /health, which vLLM serves natively (8080).
#
# Operator-controlled in the OICM UI (so NOT hard-coded here):
#   * Accelerator + slice (MIG slice vs full H200 vs multi-H200), Memory, CPU,
#     Storage/PVC size (can be raised well beyond the diagnostic's 8 GiB),
#     replicas/autoscaling, Task Type, Model Server Arguments (-> EXTRA_ARGS),
#     and arbitrary Environment Variables. This script adapts to those at runtime
#     (e.g. tensor-parallelism follows the allocation; see below).
# =============================================================================

# Abort on any error or failed pipe stage, but do NOT treat unset vars as
# errors -- OICM legitimately leaves some of its optional flags unset.
set -eo pipefail

# --- Model location (unchanged OICM contract; confirmed by the diagnostic) ---

# OICM sets INIT_CONTAINER_USE_PVC=True when its init container has populated
# the tenant PVC from object storage; the model then lives under PVC_PATH.
if [[ "${INIT_CONTAINER_USE_PVC:-}" == "True" ]]; then
    # PVC_PATH is injected as /pvc-home (the tenant's NFS mount holding weights).
    BASE_PATH=${PVC_PATH:-"/pvc-home"}
    echo "[startup] INIT_CONTAINER_USE_PVC=True -> model base: ${BASE_PATH}"
# Alternate storage mode where the model sits at the volume root instead.
elif [[ "${USE_DATA_VOLUME:-}" == "True" ]]; then
    BASE_PATH=${PVC_PATH:-"/data-volume"}
    echo "[startup] USE_DATA_VOLUME=True -> model base: ${BASE_PATH}"
# Fallback (no volume flag): use the big writable local disk, never the RO root.
else
    BASE_PATH="/tmp/oicm"
    mkdir -p "${BASE_PATH}" || { echo "[startup] FATAL: cannot create ${BASE_PATH}"; exit 1; }
    echo "[startup] no volume flag -> temporary base: ${BASE_PATH}"
fi
# Export so any child/inspection sees the resolved base.
export BASE_PATH

# OICM's download layout nests the model under <base>/app/download/base_model.
export BASE_DOWNLOAD_FOLDER="${BASE_PATH}/app/download"

# In data-volume mode the model is the volume root; otherwise it's base_model/.
if [[ "${USE_DATA_VOLUME:-}" == "True" ]]; then
    MODEL_DOWNLOAD_FOLDER="${BASE_PATH}"
else
    # Default path; OICM may override MODEL_DOWNLOAD_FOLDER directly, so honor it.
    DEFAULT_MODEL_DOWNLOAD_FOLDER="${BASE_DOWNLOAD_FOLDER}/base_model"
    MODEL_DOWNLOAD_FOLDER="${MODEL_DOWNLOAD_FOLDER:-${DEFAULT_MODEL_DOWNLOAD_FOLDER}}"
fi
echo "[startup] MODEL_DOWNLOAD_FOLDER=${MODEL_DOWNLOAD_FOLDER}"

# --- Writable HOME (because /home/runner is read-only on this platform) ---

# OICM leaves HOME=/home/runner, which is READ-ONLY; any library doing a
# ~/.something write would crash. /tmp is the big writable disk and OICM's
# XDG_* already point there, so force HOME to /tmp to catch stray ~ writers.
export HOME=/tmp

# --- Fill ONLY the cache gaps OICM left (defer to OICM where it set them) ---

# OICM set XDG_CACHE_HOME/XDG_CONFIG_HOME/HF_HOME/TRITON_CACHE_DIR/TORCH_HOME/
# FLASHINFER_WORKSPACE_BASE/TMPDIR to /tmp already, so we do NOT touch those.
#
# WHY /tmp AND NOT THE NFS PVC (this is a deliberate choice, not a size limit):
# The PVC can be sized to 80 GB+ in the UI, so capacity is NOT the reason. The
# reason is NFS SEMANTICS. The torch.compile / Triton / Inductor caches write
# thousands of tiny files using advisory locks (flock/fcntl) + atomic-rename,
# often concurrently across vLLM workers. Over NFS that means: fragile/serialized
# locking (this mount is local_lock=none), close-to-open consistency that can
# race the temp-file->rename pattern (risking stale reads / recompiles), and a
# network round-trip per small file (often slower than recompiling). /tmp is
# local ext4 -> real POSIX locking, true atomic rename, no round-trips.
# Trade-off: /tmp is ephemeral, so the compile cache is rebuilt on cold start.
# That is cheap for small models. If a large LLM ever makes warmup painful, the
# better fix is to BAKE a warmed cache into the image (restart-proof, no NFS);
# only as a last resort point VLLM_CACHE_ROOT at /pvc-home/cache via the OICM env
# field. Because everything below is ${VAR:-default}, OICM's env always wins.

# vLLM's real cache var is VLLM_CACHE_ROOT; OICM set the WRONG name
# (VLLM_CACHE_DIR), so the correct one is unset -> point it at /tmp explicitly
# instead of relying on the XDG default by luck.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm}"

# CUDA's PTX-JIT cache defaults to ~/.nv (=/home/runner/.nv, READ-ONLY). On the
# cu129-on-570 forward-compat path the JIT can fire, so give it a writable dir.
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/tmp/nv}"

# Pre-create the two dirs we introduced (the OICM /tmp ones already exist).
mkdir -p "${VLLM_CACHE_ROOT}" "${CUDA_CACHE_PATH}"

# --- Air-gap: never reach for the HuggingFace Hub (egress is blocked) ---

# Without these, HF/transformers may try the Hub and hang on the network
# timeout at startup. The model is already on disk, so offline is correct.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# --- CUDA 12.9 forward compatibility on the node's R570 (CUDA 12.8) driver ---

# The base image (vllm:0.22.0-cu129) already ships the R575 forward-compat libs
# at /usr/local/cuda/compat (verified: libcuda.so.575.57.08 + ptxjitcompiler +
# nvvm). Nothing is installed; we only need to load them AHEAD of the 570 libcuda
# that the NVIDIA container runtime injects, so the full 12.9 runtime + Triton
# JIT + CUDA-graph capture work on the older kernel driver (supported because
# the H200 is a datacenter GPU). The loop stays tolerant of a path change in a
# future base-image bump.
COMPAT_FOUND=0
for COMPAT in /usr/local/cuda/compat /usr/local/cuda-12.9/compat; do
    # Only use it if it actually contains a libcuda (avoids breaking LD path).
    if ls "${COMPAT}"/libcuda.so* >/dev/null 2>&1; then
        export LD_LIBRARY_PATH="${COMPAT}:${LD_LIBRARY_PATH:-}"
        COMPAT_FOUND=1
        echo "[startup] CUDA forward-compat enabled via ${COMPAT}"
        # Confirm the compat lib is actually first (per NVIDIA's troubleshooting).
        echo "[startup] nvidia-smi sees CUDA: $(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | head -1)"
        break
    fi
done
# Track explicitly (LD_LIBRARY_PATH may be pre-set by OICM, so don't test that).
[ "${COMPAT_FOUND}" -eq 0 ] && echo "[startup] WARNING: no cuda-compat dir found; cu129 on a 570 driver may fail at warmup."

# --- Operator-supplied extra flags (OICM's webui dev-args -> EXTRA_ARGS env) ---

# The diagnostic proved OICM passes the webui "extra args" field as EXTRA_ARGS
# (k8s args: stay empty). eval-split it into a real argv array so quoted values
# like the speculative-config JSON survive. Guarded so empty is a no-op.
EXTRA_ARGS_ARR=()
if [ -n "${EXTRA_ARGS:-}" ]; then
    eval "EXTRA_ARGS_ARR=(${EXTRA_ARGS})"
fi
# Also accept VLLM_EXTRA_ARGS as a secondary, in case you standardize on it.
if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
    eval "EXTRA_ARGS_ARR+=(${VLLM_EXTRA_ARGS})"
fi

# --- Tensor parallelism: match the allocation, but never break MIG ----------

# OICM injects NUM_GPUS for the allocation (the diagnostic showed NUM_GPUS=1 on
# a MIG slice). TP should equal the GPU count on a multi-GPU node, but MUST stay
# 1 on a MIG slice -- a MIG instance is a single isolated partition with no
# NVLink/P2P, so --tensor-parallel-size > 1 fails there. We detect MIG from the
# MIG- prefix in NVIDIA_VISIBLE_DEVICES. Skip entirely if the operator already
# set tensor parallelism (or any non-TP parallelism like -dp) via EXTRA_ARGS.
TP_ARGS=()
if [[ " ${EXTRA_ARGS:-} ${VLLM_EXTRA_ARGS:-} " != *" --tensor-parallel-size"* ]] && \
   [[ " ${EXTRA_ARGS:-} ${VLLM_EXTRA_ARGS:-} " != *" -tp"* ]] && \
   [[ " ${EXTRA_ARGS:-} ${VLLM_EXTRA_ARGS:-} " != *" --data-parallel-size"* ]] && \
   [[ " ${EXTRA_ARGS:-} ${VLLM_EXTRA_ARGS:-} " != *" -dp"* ]] && \
   [[ " ${EXTRA_ARGS:-} ${VLLM_EXTRA_ARGS:-} " != *" --pipeline-parallel-size"* ]]; then
    # MIG slice -> force single-GPU regardless of NUM_GPUS.
    if [[ "${NVIDIA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
        TP_ARGS=(--tensor-parallel-size 1)
        echo "[startup] MIG slice detected -> tensor-parallel-size=1"
    # Multiple full GPUs -> shard across them (only if NUM_GPUS is a number > 1).
    elif [[ "${NUM_GPUS:-1}" =~ ^[0-9]+$ ]] && [ "${NUM_GPUS:-1}" -gt 1 ]; then
        TP_ARGS=(--tensor-parallel-size "${NUM_GPUS}")
        echo "[startup] ${NUM_GPUS} GPUs -> tensor-parallel-size=${NUM_GPUS}"
        # TP>1 uses /dev/shm for inter-worker IPC; a small shm causes Bus errors.
        echo "[startup] NOTE: TP>1 needs adequate /dev/shm; raise it in OICM if you see 'Bus error'."
    fi
    # else: single full GPU -> leave vLLM's default of 1 (no flag needed).
fi

echo "[startup] launching vLLM (tp:${TP_ARGS[*]:-default} extra:${EXTRA_ARGS_ARR[*]:-none})"

# --- Launch -------------------------------------------------------------------

# exec so vLLM becomes PID 1's process and receives SIGTERM for clean k8s
# shutdown. Host/port are fixed (OICM's Service + /health probe expect :8080).
# --model / --served-model-name come from the OICM-injected paths and MODEL_ID.
# vLLM auto-detects the task (embed/generate/etc.) from the model's config.json,
# so no --task is set here; TP + operator args are appended last (operator wins).
exec python3 -m vllm.entrypoints.openai.api_server \
  --host=0.0.0.0 \
  --port=8080 \
  --model="${MODEL_DOWNLOAD_FOLDER}" \
  --served-model-name="${MODEL_ID}" \
  "${TP_ARGS[@]}" \
  "${EXTRA_ARGS_ARR[@]}"