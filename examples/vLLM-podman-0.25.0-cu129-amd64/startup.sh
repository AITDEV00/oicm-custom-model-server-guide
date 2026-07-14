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
#   * CUDA forward-compat is DECIDED AT RUNTIME, never assumed. The GPU fleet is
#     mixed. We probe the native driver and prepend compat ONLY when the host 
#     actually needs it.
#   * k8s liveness/readiness probes /health, which vLLM serves natively (8080).
#
# Operator-controlled in the OICM UI (so NOT hard-coded here):
#   * Accelerator + slice, Memory, CPU, Storage/PVC size.
#   * Model Server Arguments (-> EXTRA_ARGS), Environment Variables. 
# =============================================================================

# Abort on any error or failed pipe stage, but do NOT treat unset vars as errors.
set -eo pipefail

# --- 1. Base Volume Resolution ---
# Determine where OICM mounted the storage volume.
if [[ "${INIT_CONTAINER_USE_PVC:-}" == "True" ]]; then
    BASE_PATH=${PVC_PATH:-"/pvc-home"}
    echo "[startup] INIT_CONTAINER_USE_PVC=True -> model base: ${BASE_PATH}"
elif [[ "${USE_DATA_VOLUME:-}" == "True" ]]; then
    BASE_PATH=${PVC_PATH:-"/data-volume"}
    echo "[startup] USE_DATA_VOLUME=True -> model base: ${BASE_PATH}"
else
    BASE_PATH="/tmp/oicm"
    mkdir -p "${BASE_PATH}" || { echo "[startup] FATAL: cannot create ${BASE_PATH}"; exit 1; }
    echo "[startup] no volume flag -> temporary base: ${BASE_PATH}"
fi
export BASE_PATH

# --- 2. Dynamic Model Discovery (Dirty PVC & Subfolder Safe) ---
# First, check if the operator explicitly provided a valid path via environment variables
if [ -n "${MODEL_DOWNLOAD_FOLDER:-}" ] && { [ -f "${MODEL_DOWNLOAD_FOLDER}/config.json" ] || [ -f "${MODEL_DOWNLOAD_FOLDER}/params.json" ]; }; then
    echo "[startup] Operator explicitly provided valid MODEL_DOWNLOAD_FOLDER: ${MODEL_DOWNLOAD_FOLDER}"
else
    # Otherwise, safely scan for the newest model manifest (config.json or params.json).
    # This safely bypasses nested HuggingFace repository structures and strictly ignores stale/abandoned models left on reused PVCs.
    # Note: '|| true' prevents SIGPIPE from crashing the script when head closes the pipe early.
    TARGET_MANIFEST=$(find "${BASE_PATH}" -maxdepth 10 -type f \( -name "config.json" -o -name "params.json" \) -printf '%T@ %p\n' 2>/dev/null | sort -n -r | head -n 1 | cut -d' ' -f2 || true)

    if [ -z "${TARGET_MANIFEST}" ]; then
        echo "[startup] FATAL: Crawled ${BASE_PATH} (up to 10 levels) but found zero config.json or params.json files." >&2
        exit 1
    fi

    # Extract the exact directory containing the manifest to pass to vLLM
    MODEL_DOWNLOAD_FOLDER=$(dirname "${TARGET_MANIFEST}")
    echo "[startup] Discovered active model root at: ${MODEL_DOWNLOAD_FOLDER}"
fi
export MODEL_DOWNLOAD_FOLDER


# --- 3. Writable HOME ---
# OICM leaves HOME=/home/runner, which is READ-ONLY. Force to /tmp.
export HOME=/tmp

# --- 4. Cache Configurations ---
# Point vLLM and CUDA caches to local ephemeral ext4 (/tmp) to avoid NFS lock contention.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/tmp/nv}"
mkdir -p "${VLLM_CACHE_ROOT}" "${CUDA_CACHE_PATH}"

# --- 5. Air-gap Enforcements ---
# Prevent HF/transformers from trying to reach the Hub and hanging at startup.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"


# --- 6. CUDA Forward-Compatibility ---
IMAGE_CUDA_VERSION="${IMAGE_CUDA_VERSION:-12090}"
export IMAGE_CUDA_VERSION
_cuda_probe() {
    python3 - <<'PY'
import ctypes, os, sys
need = int(os.environ.get("IMAGE_CUDA_VERSION", "12090"))
try:
    lib = ctypes.CDLL("libcuda.so.1")
except OSError:
    sys.exit(201)
drv = ctypes.c_int(0)
if lib.cuDriverGetVersion(ctypes.byref(drv)) != 0:
    sys.exit(204)
sys.stderr.write("[probe] libcuda max CUDA=%d, image needs=%d\n" % (drv.value, need))
rc = lib.cuInit(0)
if rc != 0:
    sys.exit(rc if rc < 256 else 202)
if drv.value < need:
    sys.exit(10)
n = ctypes.c_int()
rc = lib.cuDeviceGetCount(ctypes.byref(n))
if rc != 0:
    sys.exit(rc if rc < 256 else 203)
sys.exit(0 if n.value > 0 else 100)
PY
}

_HOST_CUDA="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | head -1 || true)"

if _cuda_probe; then
    echo "[startup] native host driver is sufficient for the image's CUDA (${_HOST_CUDA:-version unknown}); forward-compat NOT needed -> using host libcuda directly."
else
    _probe_rc=$?
    if [ "${_probe_rc}" -eq 100 ]; then
        echo "[startup] FATAL: CUDA initialized but no GPU is visible to this pod (rc=100); check the accelerator allocation, not forward-compat." >&2
        exit 1
    fi
    echo "[startup] native libcuda not sufficient (rc=${_probe_rc}; 10=driver older than image CUDA); host driver=${_HOST_CUDA:-version unknown} -> attempting forward-compat."
    _compat_applied=0
    for COMPAT in /usr/local/cuda/compat /usr/local/cuda-12.9/compat; do
        if ls "${COMPAT}"/libcuda.so* >/dev/null 2>&1; then
            export LD_LIBRARY_PATH="${COMPAT}:${LD_LIBRARY_PATH:-}"
            _compat_applied=1
            echo "[startup] forward-compat enabled via ${COMPAT}"
            break
        fi
    done
    if [ "${_compat_applied}" -eq 0 ]; then
        echo "[startup] FATAL: native libcuda insufficient and no forward-compat dir was found; cannot start." >&2
        exit 1
    fi
    if _cuda_probe; then
        echo "[startup] forward-compat CUDA check OK -> proceeding."
    else
        _probe_rc=$?
        echo "[startup] FATAL: CUDA still not usable WITH forward-compat (rc=${_probe_rc}); driver/runtime mismatch is beyond what these compat libs can bridge." >&2
        exit 1
    fi
fi

# --- 7. Operator-supplied Extra Flags ---
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_ARGS_ARR=()
if [ -n "${EXTRA_ARGS:-}${VLLM_EXTRA_ARGS:-}" ]; then
    while IFS= read -r -d '' _tok; do
        EXTRA_ARGS_ARR+=("${_tok}")
    done < <(cd "${_SCRIPT_DIR}" && python3 -m arg_normalizer --argv-nul)
fi

# --- 8. Tensor Parallelism ---
_has_parallel_flag() {
    local t
    for t in "${EXTRA_ARGS_ARR[@]}"; do
        case "$t" in
            --tensor-parallel-size|--tensor-parallel-size=*|-tp|-tp=*|\
            --data-parallel-size|--data-parallel-size=*|-dp|-dp=*|\
            --pipeline-parallel-size|--pipeline-parallel-size=*|-pp|-pp=*)
                return 0 ;;
        esac
    done
    return 1
}

TP_ARGS=()
if ! _has_parallel_flag; then
    if [[ "${NVIDIA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
        TP_ARGS=(--tensor-parallel-size=1)
        echo "[startup] MIG slice detected -> tensor-parallel-size=1"
    elif [[ "${NUM_GPUS:-1}" =~ ^[0-9]+$ ]] && [ "${NUM_GPUS:-1}" -gt 1 ]; then
        TP_ARGS=(--tensor-parallel-size="${NUM_GPUS}")
        echo "[startup] ${NUM_GPUS} GPUs -> tensor-parallel-size=${NUM_GPUS}"
        echo "[startup] NOTE: TP>1 needs adequate /dev/shm; raise it in OICM if you see 'Bus error'."
    fi
else
    echo "[startup] operator set a parallelism flag -> leaving TP to operator args"
fi

echo "[startup] launching vLLM (tp:${TP_ARGS[*]:-default} extra:${EXTRA_ARGS_ARR[*]:-none})"

# --- 9. Launch ---
# --model comes from our dynamic discovery. --served-model-name comes from OICM.
exec python3 -m vllm.entrypoints.openai.api_server \
  --host=0.0.0.0 \
  --port=8080 \
  --model="${MODEL_DOWNLOAD_FOLDER}" \
  --served-model-name="${MODEL_ID}" \
  "${TP_ARGS[@]}" \
  "${EXTRA_ARGS_ARR[@]}"