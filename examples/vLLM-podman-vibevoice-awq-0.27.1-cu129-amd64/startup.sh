#!/bin/bash
# =============================================================================
# OICM vLLM serving entrypoint -- VibeVoice ASR (AWQ int4 split-decoder) variant.
#
# Same contract as ../vLLM-podman-0.27.1-cu129-amd64/startup.sh (env vars,
# EXTRA_ARGS handling, CUDA forward-compat probing, uid 10000 read-only rootfs),
# with VibeVoice-specific deltas:
#
#   * SPLIT-DECODER-AWARE MODEL DISCOVERY. The model dir contains TWO
#     config.json files: the root VibeVoice config AND the inner
#     decoder-awq/config.json (plain Qwen2). The base image's "newest
#     config.json wins" scan can pick the decoder (its mtime differs from the
#     root's) and silently serve a bare Qwen2 decoder. Here the root config
#     is identified by the `vibevoice_metadata` / `architectures` keys, and
#     the inner decoder config is EXCLUDED from discovery.
#
#   * MODEL_ID default + served-name validation. `--served-model-name` comes
#     from OICM's MODEL_ID exactly like the base image.
#
#   * AIR-GAP SAFE: OICM pulls the model from S3/object storage via its init
#     container onto the PVC; nothing is fetched from HF at runtime
#     (HF_HUB_OFFLINE=1 is enforced below).
#
# Operator-controlled in the OICM UI (NOT hard-coded here):
#   * Accelerator + slice, Memory, CPU, Storage/PVC size.
#   * Model Server Arguments (-> EXTRA_ARGS), Environment Variables.
# =============================================================================

set -eo pipefail

# --- 1. Base Volume Resolution (same as base image) ---
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

# --- 2. Model Discovery (VibeVoice split-decoder aware) ---
# Explicit override always wins (also how S3-pulled models with a known
# subpath should be pinned: MODEL_DOWNLOAD_FOLDER=/pvc-home/<model>).
if [ -n "${MODEL_DOWNLOAD_FOLDER:-}" ] && [ -f "${MODEL_DOWNLOAD_FOLDER}/config.json" ]; then
    echo "[startup] Operator explicitly provided valid MODEL_DOWNLOAD_FOLDER: ${MODEL_DOWNLOAD_FOLDER}"
else
    # Every config.json under the PVC, newest first.
    # For VibeVoice exports the ROOT config is the one that either declares
    # `vibevoice_metadata` (quantized split export) or has
    # architectures: ["VibeVoiceForASRTraining" | "VibeVoiceForConditionalGeneration"].
    # The inner decoder config (plain Qwen2, architectures: ["Qwen2ForCausalLM"])
    # must NEVER win discovery, otherwise we serve the bare decoder.
    MODEL_DOWNLOAD_FOLDER=""
    while IFS= read -r manifest; do
        _dir=$(dirname "${manifest}")
        if python3 - "${manifest}" <<'PYEOF'
import json, sys
try:
    cfg = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)          # unreadable/malformed -> skip candidate
archs = cfg.get("architectures") or []
if any(a.startswith("VibeVoice") for a in archs) or "vibevoice_metadata" in cfg:
    sys.exit(0)          # this is a VibeVoice root config -> accept
sys.exit(1)              # decoder / unrelated config -> skip
PYEOF
        then
            MODEL_DOWNLOAD_FOLDER="${_dir}"
            echo "[startup] Discovered VibeVoice model root at: ${MODEL_DOWNLOAD_FOLDER}"
            break
        fi
    done < <(find "${BASE_PATH}" -maxdepth 10 -type f -name "config.json" -printf '%T@ %p\n' 2>/dev/null | sort -n -r | cut -d' ' -f2)

    if [ -z "${MODEL_DOWNLOAD_FOLDER}" ]; then
        echo "[startup] FATAL: no VibeVoice root config.json found under ${BASE_PATH}." >&2
        echo "[startup]       (Set MODEL_DOWNLOAD_FOLDER explicitly if the model lives elsewhere.)" >&2
        exit 1
    fi
fi
export MODEL_DOWNLOAD_FOLDER

# --- 2b. served-model-name (OICM passes MODEL_ID) ---
MODEL_ID="${MODEL_ID:-vibevoice}"
export MODEL_ID

# --- 3. Writable HOME (rootfs is read-only under OICM) ---
export HOME=/tmp

# --- 4. Cache Configurations (node-local ext4, defer to OICM's exports) ---
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/tmp/nv}"
mkdir -p "${VLLM_CACHE_ROOT}" "${CUDA_CACHE_PATH}"

# --- 5. Air-gap Enforcements (model arrives via OICM init container / S3) ---
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# --- 5b. WSL2 pinned-memory opt-in (local dev only; no-op on cluster) ---
# Without this, vLLM on WSL2 fails at engine init with
# "RuntimeError: UVA is not available". Harmless/unset on cluster GPU nodes.
export VLLM_WSL2_ENABLE_PIN_MEMORY="${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}"

# --- 6. CUDA Forward-Compatibility (identical to base image) ---
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

# --- 7. Operator-supplied Extra Flags (EXTRA_ARGS via arg_normalizer) ---
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_ARGS_ARR=()
if [ -n "${EXTRA_ARGS:-}${VLLM_EXTRA_ARGS:-}" ]; then
    while IFS= read -r -d '' _tok; do
        EXTRA_ARGS_ARR+=("${_tok}")
    done < <(cd "${_SCRIPT_DIR}" && python3 -m arg_normalizer --argv-nul)
fi

# --- 7b. VibeVoice safety defaults ---
# trust-remote-code is REQUIRED (custom architecture); inject if missing so an
# operator cannot accidentally boot into "unknown architecture" errors.
_has_tr() {
    local t
    for t in "${EXTRA_ARGS_ARR[@]}"; do
        case "$t" in
            --trust-remote-code|--trust-remote-code=true) return 0 ;;
        esac
    done
    return 1
}
if ! _has_tr; then
    EXTRA_ARGS_ARR+=(--trust-remote-code)
    echo "[startup] injected required --trust-remote-code"
fi

# AWQ int4 checkpoints must run in fp16; default to it unless the operator set
# --dtype explicitly (vLLM argparse last-wins keeps operator override working).
_has_dtype() {
    local t
    for t in "${EXTRA_ARGS_ARR[@]}"; do
        case "$t" in
            --dtype|--dtype=*) return 0 ;;
        esac
    done
    return 1
}
if ! _has_dtype; then
    EXTRA_ARGS_ARR+=(--dtype float16)
    echo "[startup] no --dtype given -> defaulting to float16 (required for AWQ int4)"
fi

# --- 8. Tensor Parallelism (same policy as base image) ---
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
# --model comes from split-aware discovery; --served-model-name from OICM.
exec python3 -m vllm.entrypoints.openai.api_server \
  --host=0.0.0.0 \
  --port="${PORT:-8080}" \
  --model="${MODEL_DOWNLOAD_FOLDER}" \
  --served-model-name="${MODEL_ID}" \
  "${TP_ARGS[@]}" \
  "${EXTRA_ARGS_ARR[@]}"
