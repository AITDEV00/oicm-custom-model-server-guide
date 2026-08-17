#!/bin/bash
# =============================================================================
# OICM SGLang serving entrypoint -- hardened against the REAL OICM runtime.
# Grounded in the live diagnostic + verified follow-ups (mirrors the vLLM
# podman variant; adjusted for SGLang's CLI and env vars):
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
#     mixed: H200 nodes run R580 / CUDA 13.0 and support the cu129 image NATIVELY,
#     while older nodes may be on R570 / CUDA 12.8 and need the forward-compat libs
#     the base image ships at /usr/local/cuda/compat. We probe the native driver
#     and prepend compat ONLY when the host actually needs it -- forcing compat on
#     a driver that is already new enough loads an OLDER user-mode libcuda than the
#     kernel module and triggers CUDA error 803 at init. See the compat section.
#   * k8s liveness/readiness probes /health, which SGLang serves natively (8080).
#
# Operator-controlled in the OICM UI (so NOT hard-coded here):
#   * Accelerator + slice (MIG slice vs full H200 vs multi-H200), Memory, CPU,
#     Storage/PVC size (can be raised well beyond the diagnostic's 8 GiB),
#     replicas/autoscaling, Task Type, Model Server Arguments (-> EXTRA_ARGS),
#     and arbitrary Environment Variables. This script adapts to those at runtime
#     (e.g. tensor-parallelism follows the allocation; see below).
# =============================================================================

# Abort on any error or failed pipe stage, but do not treat unset vars as
# errors -- OICM legitimately leaves some of its optional flags unset.
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
if [ -n "${MODEL_DOWNLOAD_FOLDER:-}" ] && { [ -f "${MODEL_DOWNLOAD_FOLDER}/config.json" ] || [ -f "${MODEL_DOWNLOAD_FOLDER}/params.json" ] || [ -f "${MODEL_DOWNLOAD_FOLDER}/model_index.json" ]; }; then
    # Capture which manifest type the operator's path contains, so section 9
    # can route to the correct SGLang server (diffusion vs LLM/embedding).
    if [ -f "${MODEL_DOWNLOAD_FOLDER}/model_index.json" ]; then
        MANIFEST_BASENAME="model_index.json"
    elif [ -f "${MODEL_DOWNLOAD_FOLDER}/config.json" ]; then
        MANIFEST_BASENAME="config.json"
    else
        MANIFEST_BASENAME="params.json"
    fi
    export MANIFEST_BASENAME
    echo "[startup] Operator explicitly provided valid MODEL_DOWNLOAD_FOLDER: ${MODEL_DOWNLOAD_FOLDER} (manifest: ${MANIFEST_BASENAME})"
else
    # Otherwise, safely scan for the model root manifest.
    #
    # Recognized root manifests:
    #   config.json      -- HuggingFace transformers models (has "model_type")
    #   model_index.json -- diffusers models (ideogram, SDXL, Flux, etc.)
    #   params.json      -- Mistral/GGUF-style models
    #
    # CRITICAL: sort by SHALLOWEST depth first, NOT newest mtime. A diffusers
    # model root has model_index.json (not config.json), and its component
    # subfolders (vae/, transformer/, scheduler/) each have their OWN config.json
    # -- those are component configs, NOT the main model manifest. If we sorted by
    # mtime, a freshly-touched vae/config.json would win and SGLang would try to
    # load the VAE as the main model ("Should have a `model_type` key"). The model
    # root is always at a SHALLOWER depth than its components, so depth-first
    # sorting picks the root manifest and stops before crawling into the model's
    # internal directory structure. Mtime is only a tiebreaker (newest first) for
    # the case where multiple models coexist on a reused PVC at the same depth.
    #
    # CAUTION for speculative-decoding deployments (EAGLE/MTP roadmap): a draft
    # model is a separate model dir with its OWN config.json, pointed to by
    # --speculative-draft-model-path. If the target and draft models share a PVC
    # tree at the same depth, this heuristic can still pick the wrong one. For any
    # speculative-decoding deployment, have OICM set MODEL_DOWNLOAD_FOLDER
    # explicitly so this discovery is bypassed entirely.
    #
    # Note: '|| true' prevents SIGPIPE from crashing the script when head closes
    # the pipe early. find -printf '%d' = depth from start point; sort -k1,1n
    # = depth ascending, -k2,2nr = mtime descending as tiebreaker.
    TARGET_MANIFEST=$(find "${BASE_PATH}" -maxdepth 10 -type f \( -name "config.json" -o -name "params.json" -o -name "model_index.json" \) -printf '%d %T@ %p\n' 2>/dev/null | sort -k1,1n -k2,2nr | head -n 1 | cut -d' ' -f3 || true)

    if [ -z "${TARGET_MANIFEST}" ]; then
        echo "[startup] FATAL: Crawled ${BASE_PATH} (up to 10 levels) but found zero config.json or params.json files." >&2
        exit 1
    fi

    # Extract the exact directory containing the manifest to pass to SGLang
    MODEL_DOWNLOAD_FOLDER=$(dirname "${TARGET_MANIFEST}")
    # Capture which manifest type was found, so section 9 can route to the
    # correct SGLang server: model_index.json -> diffusion (sglang serve),
    # config.json/params.json -> LLM/embedding (python3 -m sglang.launch_server).
    MANIFEST_BASENAME="$(basename "${TARGET_MANIFEST}")"
    export MODEL_DOWNLOAD_FOLDER MANIFEST_BASENAME
    echo "[startup] Discovered active model root at: ${MODEL_DOWNLOAD_FOLDER} (manifest: ${MANIFEST_BASENAME})"
fi
export MODEL_DOWNLOAD_FOLDER


# --- 3. Writable HOME + TMPDIR ---
# OICM leaves HOME=/home/runner, which is READ-ONLY. Force to /tmp.
export HOME=/tmp

# Pin TMPDIR defensively. SGLang's inter-process ZMQ sockets are AF_UNIX files
# built via tempfile.NamedTemporaryFile() (PortArgs.init_new: scheduler_input_ipc,
# tokenizer_ipc, detokenizer_ipc, rpc_ipc, metrics_ipc) -- all honor TMPDIR, then
# fall back to /tmp. OICM exports TMPDIR=/tmp already, but a cluster variant that
# doesn't would leave socket creation to Python's default (still /tmp, writable).
# Pinning removes the ambiguity and keeps every transient write on local ext4.
export TMPDIR="${TMPDIR:-/tmp}"

# --- 4. Cache Configurations ---
# Point SGLang and CUDA caches to local ephemeral ext4 (/tmp) to avoid NFS lock
# contention. Verified against the SGLang 0.5.17 source:
#   * SGLANG_CACHE_DIR (default ~/.cache/sglang) is SGLang's own cache root --
#     initialize_cache() creates inductor_cache/ + triton_cache/ subdirs under it
#     and sets TORCHINDUCTOR_CACHE_DIR/TRITON_CACHE_DIR itself. Pinning it to /tmp
#     makes the whole tree (incl. the JIT kernel cache) land predictably.
#   * SGLANG_DG_CACHE_DIR (default ~/.cache/deep_gemm) holds the DeepGEMM JIT
#     cache, which fires on Hopper (H200 = SM90) at startup.
#   * TORCHINDUCTOR_CACHE_DIR is the REAL torch.compile cache var (PyTorch's, not
#     a SGLANG_* name -- there is no SGLANG_TORCH_COMPILE_CACHE_DIR in the source).
#     Set explicitly as defense-in-depth; SGLang relocates it under SGLANG_CACHE_DIR.
#   * CUDA_CACHE_PATH is NVIDIA's PTX-JIT cache var (genuine NVIDIA env var).
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-/tmp/sglang}"
export SGLANG_DG_CACHE_DIR="${SGLANG_DG_CACHE_DIR:-/tmp/sglang/deep_gemm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/sglang/inductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/tmp/nv}"
mkdir -p "${SGLANG_CACHE_DIR}" "${SGLANG_DG_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${CUDA_CACHE_PATH}"

# Prometheus multiprocess dir for --enable-metrics. SGLang uses multiprocess
# collection (tokenizer/scheduler/detokenizer are separate processes), so the
# /metrics handler calls prometheus_client.multiprocess.MultiProcessCollector,
# which requires a writable PROMETHEUS_MULTIPROC_DIR. Verified in 0.5.17
# source: set_prometheus_multiproc_dir() (utils/common.py:1550) DOES handle the
# unset case itself -- it calls tempfile.TemporaryDirectory() (no dir=), which
# lands under TMPDIR -> /tmp. So the default is safe. We still set it explicitly
# to (a) make the location predictable and co-located with the other SGLang
# caches, and (b) guard against a cluster variant where TMPDIR points somewhere
# odd.
# When set, SGLang creates a sub-TemporaryDirectory *inside* this dir, so it
# must exist and be writable -- hence the mkdir. Set BEFORE the server imports
# prometheus_client (it does, via add_prometheus_middleware at startup).
export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-/tmp/sglang/prom}"
mkdir -p "${PROMETHEUS_MULTIPROC_DIR}"

# --- 5. Air-gap Enforcements ---
# Prevent HF/transformers from trying to reach the Hub and hanging at startup.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"


# --- 6. CUDA Forward-Compatibility ---
# CUDA the IMAGE was built for, encoded as 1000*major + 10*minor (cu129 -> 12090).
# This is a property of THIS image's base (FROM ...cu129...), not a node guess;
# bump it only when the base image's CUDA changes.
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
# OICM "Model Server Arguments" -> EXTRA_ARGS (and SGLANG_EXTRA_ARGS as an alias).
# /app/arg_normalizer.py accepts every form an operator might type (=/space/
# quoted/unquoted JSON/scalar/dot-notation), auto-protects unquoted JSON values,
# folds space form into = form, and dedups repeated flags last-wins. Pure stdlib.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_ARGS_ARR=()
if [ -n "${EXTRA_ARGS:-}${SGLANG_EXTRA_ARGS:-}" ]; then
    while IFS= read -r -d '' _tok; do
        EXTRA_ARGS_ARR+=("${_tok}")
    done < <(cd "${_SCRIPT_DIR}" && python3 -m arg_normalizer --argv-nul)
fi

# --- 8. Tensor Parallelism ---
# SGLang has TWO servers with DISJOINT parallelism flags:
#   * launch_server (LLM/embedding):  --tp-size / --tensor-parallel-size (alias)
#     --dp-size / --data-parallel-size (alias)
#     --pp-size / --pipeline-parallel-size (alias)   -- NOTE: bare --pp is REJECTED
#     ("ambiguous option" with --pp-max-micro-batch-size /
#     --pp-async-batch-depth), so never use the short form here.
#     --ep-size / --expert-parallel-size / --ep      (MoE expert parallelism)
#   * sglang serve (diffusion):       --num-gpus  (no --tp-size here!)
#     --ulysses-degree, --ring-degree  (sequence-parallel for attention)
# Scan the NORMALIZED tokens so =, space, and short forms are all covered. The
# DP case is the dangerous one: if an operator passes only --dp-size, injecting
# --tp-size on top would create a TP x DP allocation the node can't satisfy, so
# any parallel flag disables the auto-injection entirely.
_has_parallel_flag() {
    local t
    for t in "${EXTRA_ARGS_ARR[@]}"; do
        case "$t" in
            --tp-size|--tp-size=*|\
            --tensor-parallel-size|--tensor-parallel-size=*|\
            --dp-size|--dp-size=*|\
            --data-parallel-size|--data-parallel-size=*|\
            --pp-size|--pp-size=*|\
            --pipeline-parallel-size|--pipeline-parallel-size=*|\
            --ep-size|--ep-size=*|\
            --expert-parallel-size|--expert-parallel-size=*|--ep|--ep=*|\
            --num-gpus|--num-gpus=*|\
            --ulysses-degree|--ulysses-degree=*|\
            --ring-degree|--ring-degree=*)
                return 0 ;;
        esac
    done
    return 1
}

TP_ARGS=()
if ! _has_parallel_flag; then
    if [[ "${NVIDIA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
        # MIG slice = 1 GPU, regardless of server type. Choose the correct
        # parallelism flag for the model class: --num-gpus for diffusion
        # (sglang serve), --tp-size for LLM/embedding (launch_server).
        if [ "${MANIFEST_BASENAME}" = "model_index.json" ]; then
            TP_ARGS=(--num-gpus=1)
            echo "[startup] MIG slice + diffusion model -> num-gpus=1"
        else
            TP_ARGS=(--tp-size=1)
            echo "[startup] MIG slice + LLM/embedding model -> tp-size=1"
        fi
    elif [[ "${NUM_GPUS:-1}" =~ ^[0-9]+$ ]] && [ "${NUM_GPUS:-1}" -gt 1 ]; then
        if [ "${MANIFEST_BASENAME}" = "model_index.json" ]; then
            TP_ARGS=(--num-gpus="${NUM_GPUS}")
            echo "[startup] ${NUM_GPUS} GPUs + diffusion model -> num-gpus=${NUM_GPUS}"
        else
            TP_ARGS=(--tp-size="${NUM_GPUS}")
            echo "[startup] ${NUM_GPUS} GPUs + LLM/embedding model -> tp-size=${NUM_GPUS}"
        fi
        echo "[startup] NOTE: multi-GPU needs adequate /dev/shm; raise it in OICM if you see 'Bus error'."
    fi
else
    echo "[startup] operator set a parallelism flag -> leaving parallelism to operator args"
fi

echo "[startup] launching SGLang (tp:${TP_ARGS[*]:-default} extra:${EXTRA_ARGS_ARR[*]:-none})"

# --- 8a. Debug Dump ---
# Comprehensive environment + state dump for post-mortem debugging. This fires
# BEFORE the exec, so even in CrashLoopBackOff the logs (kubectl logs --previous)
# contain the full picture: who we are, what we see, what we computed, and what
# we're about to launch. No kubectl exec needed. Secrets are redacted.
# Controlled by DEBUG_DUMP (default: on; set DEBUG_DUMP=0 to silence).
if [[ "${DEBUG_DUMP:-1}" != "0" ]]; then
    echo "[debug] ═══════════════════════════════════════════════════════════════"
    echo "[debug] STARTUP DEBUG DUMP @ $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

    # --- 8a.1 Identity & Process ---
    echo "[debug] ── Identity ──"
    echo "[debug]   whoami=$(whoami 2>/dev/null || echo 'n/a')  uid=$(id -u)  gid=$(id -g)"
    echo "[debug]   groups=$(id 2>/dev/null || echo 'n/a')"
    echo "[debug]   hostname=$(hostname 2>/dev/null || echo 'n/a')"
    echo "[debug]   pwd=$(pwd)"
    echo "[debug]   script=${BASH_SOURCE[0]:-startup.sh}"
    echo "[debug]   bash=${BASH_VERSION:-unknown}"
    echo "[debug]   pid=$$  ppid=$PPID"

    # --- 8a.2 Environment Variables (redacted) ---
    # Print ALL env vars sorted, but redact values that look like secrets
    # (keys matching KEY|SECRET|TOKEN|PASSWORD|PASS|CREDENTIAL|ACCESS_KEY).
    echo "[debug] ── Environment Variables (secrets redacted) ──"
    if command -v env >/dev/null 2>&1; then
        env 2>/dev/null | sort | while IFS='=' read -r _k _v; do
            case "${_k}" in
                *KEY*|*SECRET*|*TOKEN*|*PASSWORD*|*PASS*|*CREDENTIAL*|*ACCESS_KEY*)
                    echo "[debug]   ${_k}=<REDACTED>" ;;
                *)
                    echo "[debug]   ${_k}=${_v}" ;;
            esac
        done
    else
        # Fallback: print from /proc/self/environ if env is unavailable
        if [ -r /proc/self/environ ]; then
            tr '\0' '\n' < /proc/self/environ | sort | while IFS='=' read -r _k _v; do
                case "${_k}" in
                    *KEY*|*SECRET*|*TOKEN*|*PASSWORD*|*PASS*|*CREDENTIAL*|*ACCESS_KEY*)
                        echo "[debug]   ${_k}=<REDACTED>" ;;
                    *)
                        echo "[debug]   ${_k}=${_v}" ;;
                esac
            done
        else
            echo "[debug]   (env and /proc/self/environ both unavailable)"
        fi
    fi

    # --- 8a.3 Computed Startup State ---
    echo "[debug] ── Computed State ──"
    echo "[debug]   BASE_PATH=${BASE_PATH:-<unset>}"
    echo "[debug]   MODEL_DOWNLOAD_FOLDER=${MODEL_DOWNLOAD_FOLDER:-<unset>}"
    echo "[debug]   MANIFEST_BASENAME=${MANIFEST_BASENAME:-<unset>}"
    echo "[debug]   MODEL_ID=${MODEL_ID:-<unset>}"
    echo "[debug]   MODEL_SOURCE=${MODEL_SOURCE:-<unset>}"
    echo "[debug]   INIT_CONTAINER_USE_PVC=${INIT_CONTAINER_USE_PVC:-<unset>}"
    echo "[debug]   USE_DATA_VOLUME=${USE_DATA_VOLUME:-<unset>}"
    echo "[debug]   PVC_PATH=${PVC_PATH:-<unset>}"
    echo "[debug]   NUM_GPUS=${NUM_GPUS:-<unset>}"
    echo "[debug]   NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
    echo "[debug]   EXTRA_ARGS=${EXTRA_ARGS:-<unset>}"
    echo "[debug]   SGLANG_EXTRA_ARGS=${SGLANG_EXTRA_ARGS:-<unset>}"
    echo "[debug]   HOME=${HOME}"
    echo "[debug]   TMPDIR=${TMPDIR}"
    echo "[debug]   LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
    echo "[debug]   IMAGE_CUDA_VERSION=${IMAGE_CUDA_VERSION}"
    echo "[debug]   SGLANG_CACHE_DIR=${SGLANG_CACHE_DIR}"
    echo "[debug]   SGLANG_DG_CACHE_DIR=${SGLANG_DG_CACHE_DIR}"
    echo "[debug]   TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"
    echo "[debug]   CUDA_CACHE_PATH=${CUDA_CACHE_PATH}"
    echo "[debug]   PROMETHEUS_MULTIPROC_DIR=${PROMETHEUS_MULTIPROC_DIR}"
    echo "[debug]   HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
    echo "[debug]   TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
    echo "[debug]   TP_ARGS=[${TP_ARGS[*]:-empty}]"
    echo "[debug]   SERVED_ARGS=[${SERVED_ARGS[*]:-empty}]"
    echo "[debug]   EXTRA_ARGS_ARR=[${EXTRA_ARGS_ARR[*]:-empty}]"
    echo "[debug]   LORA_ADAPTER_ARGS=${LORA_ADAPTER_ARGS:-<unset>}"
    echo "[debug]   LORA_ADAPTER_SOURCES=${LORA_ADAPTER_SOURCES:-<unset>}"

    # --- 8a.4 GPU & CUDA ---
    echo "[debug] ── GPU & CUDA ──"
    echo "[debug]   nvidia-smi (first 40 lines):"
    nvidia-smi 2>/dev/null | head -40 | sed 's/^/[debug]     /' || echo "[debug]     nvidia-smi not available"
    echo "[debug]   CUDA Toolkit:"
    if [ -f /usr/local/cuda/version.json ]; then
        python3 -c "import json; v=json.load(open('/usr/local/cuda/version.json')); print('  build:', v.get('build', '?'), '  version:', v.get('cuda_version', v.get('version', '?')))" 2>/dev/null | sed 's/^/[debug]   /' || echo "[debug]     (version.json parse failed)"
    elif command -v nvcc >/dev/null 2>&1; then
        nvcc --version 2>/dev/null | sed 's/^/[debug]     /' || echo "[debug]     nvcc --version failed"
    else
        echo "[debug]     nvcc not found"
    fi
    echo "[debug]   libcuda.so locations:"
    ldconfig -p 2>/dev/null | grep -i 'libcuda' | sed 's/^/[debug]     /' || echo "[debug]     (ldconfig not available)"
    echo "[debug]   CUDA forward-compat dirs:"
    for _d in /usr/local/cuda/compat /usr/local/cuda-12.9/compat; do
        if [ -d "${_d}" ]; then
            echo "[debug]     ${_d}/ : $(ls "${_d}" 2>/dev/null | tr '\n' ' ')"
        else
            echo "[debug]     ${_d}/ : <not found>"
        fi
    done

    # --- 8a.5 Model Files ---
    echo "[debug] ── Model Directory ──"
    if [ -n "${MODEL_DOWNLOAD_FOLDER:-}" ] && [ -d "${MODEL_DOWNLOAD_FOLDER}" ]; then
        echo "[debug]   path: ${MODEL_DOWNLOAD_FOLDER}"
        echo "[debug]   top-level contents:"
        ls -lah "${MODEL_DOWNLOAD_FOLDER}" 2>/dev/null | head -40 | sed 's/^/[debug]     /' || echo "[debug]     (ls failed)"
        echo "[debug]   manifest (${MANIFEST_BASENAME}):"
        if [ -f "${MODEL_DOWNLOAD_FOLDER}/${MANIFEST_BASENAME}" ]; then
            head -50 "${MODEL_DOWNLOAD_FOLDER}/${MANIFEST_BASENAME}" 2>/dev/null | sed 's/^/[debug]     /' || echo "[debug]     (read failed)"
        else
            echo "[debug]     <manifest not found at expected path>"
        fi
        echo "[debug]   total size:"
        du -sh "${MODEL_DOWNLOAD_FOLDER}" 2>/dev/null | sed 's/^/[debug]     /' || echo "[debug]     (du failed)"
        echo "[debug]   subdir tree (2 levels):"
        find "${MODEL_DOWNLOAD_FOLDER}" -maxdepth 2 -type f 2>/dev/null | head -60 | sed 's/^/[debug]     /' || echo "[debug]     (find failed)"
    else
        echo "[debug]   <MODEL_DOWNLOAD_FOLDER not set or not a directory>"
    fi

    # --- 8a.6 PVC / Base Path Tree ---
    echo "[debug] ── Base Path Tree (2 levels) ──"
    if [ -d "${BASE_PATH:-}" ]; then
        find "${BASE_PATH}" -maxdepth 2 2>/dev/null | head -40 | sed 's/^/[debug]     /' || echo "[debug]     (find failed)"
    else
        echo "[debug]   <BASE_PATH not set or not a directory>"
    fi

    # --- 8a.7 Disk & Memory ---
    echo "[debug] ── Disk & Memory ──"
    echo "[debug]   df (key mounts):"
    df -h / /tmp /dev/shm "${BASE_PATH:-/pvc-home}" 2>/dev/null | sed 's/^/[debug]     /' || echo "[debug]     (df failed)"
    echo "[debug]   /dev/shm:"
    ls -lah /dev/shm 2>/dev/null | head -10 | sed 's/^/[debug]     /' || echo "[debug]     (not accessible)"
    echo "[debug]   memory:"
    free -h 2>/dev/null | sed 's/^/[debug]     /' || echo "[debug]     (free not available)"

    # --- 8a.8 SGLang / Python Versions ---
    echo "[debug] ── Software Versions ──"
    echo "[debug]   python3: $(python3 --version 2>&1 || echo 'not found')"
    echo "[debug]   sglang CLI: $(sglang --version 2>&1 || echo 'sglang CLI not found')"
    echo "[debug]   sglang module: $(python3 -c 'import sglang; print(sglang.__version__)' 2>&1 || echo 'import failed')"
    echo "[debug]   torch: $(python3 -c 'import torch; print(torch.__version__, "CUDA:", torch.version.cuda, "available:", torch.cuda.is_available())' 2>&1 || echo 'import failed')"
    echo "[debug]   diffusers: $(python3 -c 'import diffusers; print(diffusers.__version__)' 2>&1 || echo 'NOT INSTALLED')"
    echo "[debug]   transformers: $(python3 -c 'import transformers; print(transformers.__version__)' 2>&1 || echo 'NOT INSTALLED')"
    echo "[debug]   bitsandbytes: $(python3 -c 'import bitsandbytes; print(bitsandbytes.__version__)' 2>&1 || echo 'NOT INSTALLED')  # only for MIG-slice diffusion (ideogram-4-nf4); removed from this image"
    echo "[debug]   triton: $(python3 -c 'import triton; print(triton.__version__)' 2>&1 || echo 'NOT INSTALLED')"
    echo "[debug]   fastvideo: $(python3 -c 'import fastvideo; print(fastvideo.__version__)' 2>&1 || echo 'NOT INSTALLED')"
    echo "[debug]   prometheus_client: $(python3 -c 'import prometheus_client; print(prometheus_client.__version__)' 2>&1 || echo 'NOT INSTALLED')"
    echo "[debug]   vllm (if coinstalled): $(python3 -c 'import vllm; print(vllm.__version__)' 2>&1 || echo 'NOT INSTALLED')"
    echo "[debug]   flashinfer: $(python3 -c 'import flashinfer; print(flashinfer.__version__)' 2>&1 || echo 'NOT INSTALLED')"

    # --- 8a.9 SGLang Serve --help (diffusion server flags) ---
    echo "[debug] ── sglang serve --help (FULL output, for flag reference) ──"
    sglang serve --help 2>&1 | sed 's/^/[debug]     /' || echo "[debug]     (sglang serve --help failed)"

    # --- 8a.10 launch_server --help (LLM server flags) ---
    echo "[debug] ── python3 -m sglang.launch_server --help (FULL output, for flag reference) ──"
    python3 -m sglang.launch_server --help 2>&1 | sed 's/^/[debug]     /' || echo "[debug]     (launch_server --help failed)"

    # --- 8a.11 Cache Dirs Writability ---
    echo "[debug] ── Cache Dir Writability ──"
    for _d in "${SGLANG_CACHE_DIR}" "${SGLANG_DG_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${PROMETHEUS_MULTIPROC_DIR}" "${HOME}" "${TMPDIR}"; do
        if [ -d "${_d}" ]; then
            if [ -w "${_d}" ]; then
                echo "[debug]   ${_d} : WRITABLE"
            else
                echo "[debug]   ${_d} : NOT WRITABLE"
            fi
        else
            echo "[debug]   ${_d} : DOES NOT EXIST"
        fi
    done

    echo "[debug] ═══════════════════════════════════════════════════════════════"
fi

# --- 9. Launch ---
# SGLang has TWO separate servers with DISJOINT flag sets and model classes.
# The manifest type discovered in section 2 determines which server to use:
#
#   * model_index.json  -> DIFFUSION model (ideogram, FLUX, Wan, etc.)
#     Use `sglang serve` (the diffusion server). It does NOT accept
#     --enable-metrics or --served-model-name (those are LLM-only flags that
#     live in launch_server's argparser). Parallelism is --num-gpus (not
#     --tp-size). The model is identified in API requests by its model-path,
#     not a served-name flag. Endpoints: OpenAI-compatible /v1/images and
#     /v1/videos APIs.
#
#   * config.json / params.json -> LLM / embedding model (Qwen, MiniMax, etc.)
#     Use `python3 -m sglang.launch_server` (the LLM server). This is the
#     verified-correct entrypoint for LLMs -- it accepts --enable-metrics,
#     --served-model-name, --tp-size, --mem-fraction-static, etc. Endpoints:
#     /v1/chat/completions, /v1/embeddings, /health, /metrics.
#
# `sglang serve` is NOT a superset of launch_server (the earlier comment was
# wrong). It is a different server for a different model class. Mixing the two
# sends LLM flags into the diffusion server (or vice versa) and SGLang's
# argparser rejects them -- exactly the crash we saw with ideogram-4-nf4.
#
# --served-model-name is OMITTED when MODEL_ID is unset (we're on `set -eo`, not
# `-u`, so an unset MODEL_ID would expand to `--served-model-name=` -- an empty
# string that registers the model under "" and makes every client request with a
# real model name get rejected). SGLang defaults the served name to the model
# path when the flag is absent, which is the correct fallback. Build the arg
# conditionally and splice the array into the exec line (LLM path only).
#
# DIFFUSION uses --model-id (NOT --served-model-name). The diffusion server's
# argparser does not have --served-model-name; it has --model-id, described as:
#   "Override the model ID used for config resolution. Useful when --model-path
#    is a local directory whose name does not match any registered HF repo name.
#    Should be the repo name portion of the HF ID (e.g. 'Qwen-Image' for
#    'Qwen/Qwen-Image')."
# This matters because OICM downloads the model to /pvc-home/app/download/base_model
# (a generic path), so the diffusion server can't infer the HF repo name from the
# path. Passing --model-id=ideogram-ai/ideogram-4-nf4 lets it resolve the correct
# pipeline config, tokenizer, and chat template.
SERVED_ARGS=()
if [ -n "${MODEL_ID:-}" ]; then
    if [ "${MANIFEST_BASENAME}" = "model_index.json" ]; then
        SERVED_ARGS=(--model-id="${MODEL_ID}")
    else
        SERVED_ARGS=(--served-model-name="${MODEL_ID}")
    fi
else
    echo "[startup] MODEL_ID unset -> omitting model name flag (SGLang defaults to model path)"
fi

# Build the full argv into one array so we can log the EXACT command before exec.
# (Quote-print each token so a value with spaces/quotes is visible -- rare for
# flags, but the JSON-valued ones like --json-model-override-args={...} benefit.)
if [ "${MANIFEST_BASENAME}" = "model_index.json" ]; then
    # --- DIFFUSION path (sglang serve) ---
    # No --enable-metrics (LLM-only). The diffusion server uses --model-id
    # (not --served-model-name) for model identification -- see SERVED_ARGS
    # above. --num-gpus comes from section 8 (NOT --tp-size, which the
    # diffusion server rejects).
    #
    # --output-path defaults to "outputs/" (relative to CWD=/app, which is
    # READ-ONLY under OICM). The image API's temp_dir_if_disabled() calls
    # os.makedirs(configured_path) on every request, crashing with
    # "OSError: [Errno 30] Read-only file system: 'outputs/'". Point it to
    # /tmp/sglang/outputs (writable local ext4). The API still returns
    # b64_json in the response -- this dir is only used when the server
    # needs to temporarily write image files during generation.
    DIFFUSION_OUTPUT_PATH="${DIFFUSION_OUTPUT_PATH:-/tmp/sglang/outputs}"
    mkdir -p "${DIFFUSION_OUTPUT_PATH}"

    LAUNCH_ARGS=(
      --host=0.0.0.0
      --port=8080
      --model-path="${MODEL_DOWNLOAD_FOLDER}"
      --output-path="${DIFFUSION_OUTPUT_PATH}"
    )
    LAUNCH_ARGS+=("${SERVED_ARGS[@]}")
    LAUNCH_ARGS+=("${TP_ARGS[@]}")
    LAUNCH_ARGS+=("${EXTRA_ARGS_ARR[@]}")

    echo "[startup] model class: DIFFUSION (manifest=${MANIFEST_BASENAME}) -> using 'sglang serve'"
    echo "[startup] ===== exec command ====="
    printf '  %q\n' sglang serve "${LAUNCH_ARGS[@]}"
    echo "[startup] ----- on one line -----"
    # shellcheck disable=SC2086
    echo "[startup] $ sglang serve ${LAUNCH_ARGS[*]}"
    echo "[startup] ========================="

    exec sglang serve "${LAUNCH_ARGS[@]}"
else
    # --- LLM / embedding path (python3 -m sglang.launch_server) ---
    # Full flag set: --enable-metrics for /metrics scrape, --served-model-name
    # for model routing, --tp-size for tensor parallelism. All verified valid
    # in launch_server's argparser on 0.5.17.
    LAUNCH_ARGS=(
      --host=0.0.0.0
      --port=8080
      --model-path="${MODEL_DOWNLOAD_FOLDER}"
      --enable-metrics
    )
    LAUNCH_ARGS+=("${SERVED_ARGS[@]}")
    LAUNCH_ARGS+=("${TP_ARGS[@]}")
    LAUNCH_ARGS+=("${EXTRA_ARGS_ARR[@]}")

    echo "[startup] model class: LLM/EMBEDDING (manifest=${MANIFEST_BASENAME}) -> using 'python3 -m sglang.launch_server'"
    echo "[startup] ===== exec command ====="
    printf '  %q\n' python3 -m sglang.launch_server "${LAUNCH_ARGS[@]}"
    echo "[startup] ----- on one line -----"
    # shellcheck disable=SC2086
    echo "[startup] $ python3 -m sglang.launch_server ${LAUNCH_ARGS[*]}"
    echo "[startup] ========================="

    exec python3 -m sglang.launch_server "${LAUNCH_ARGS[@]}"
fi
