#!/bin/bash
# =============================================================================
# OICM SGLang-Omni serving entrypoint -- hardened against the REAL OICM runtime.
#
# Mirrors the SGLang harness (same sections, same field-proven CUDA-compat
# probe), but launches the Omni pipeline server instead of sglang:
#
#   sgl-omni serve --host 0.0.0.0 --port 8080 --model-path <dir>
#
# Fixed by OICM (so handled here, in the image):
#   * Runs as uid 10000 / gid 0, HOME=/home/runner -> OICM RESPECTS the image
#     UID, but / and /home/runner are READ-ONLY.
#   * Root filesystem is read-only. /tmp is a large, writable, NODE-LOCAL ext4
#     disk (ephemeral: wiped on pod restart/reschedule). The model arrives on
#     the NFS PVC at /pvc-home via OICM's init container.
#   * OICM already exports HF_HOME, TRITON_CACHE_DIR, TORCH_HOME,
#     XDG_CACHE_HOME, TMPDIR -> /tmp. We DEFER to those and only fill the gaps.
#   * Cluster is air-gapped (HF / PyPI unreachable) -> must run fully offline.
#   * k8s liveness/readiness probes /health, which sgl-omni serves natively
#     (200 healthy, 503 while starting). It ALSO serves /v1/models.
#   * CUDA forward-compat is DECIDED AT RUNTIME, never assumed. See section 6.
#   * /dev/shm: sgl-omni relays large frames through POSIX shared memory.
#     OICM gives 5 GiB by default; raise it if you see "Bus error".
#
# Operator-controlled in the OICM UI (so NOT hard-coded here):
#   * Accelerator + slice, Memory, CPU, Storage/PVC size, replicas/autoscaling,
#     Task Type, Model Server Arguments (-> EXTRA_ARGS), Environment Variables.
# =============================================================================

# Abort on any error or failed pipe stage, but do not treat unset vars as
# errors -- OICM legitimately leaves some of its optional flags unset.
set -eo pipefail

# --- 1. Base Volume Resolution ---
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
# SGLang-Omni dispatches on the HF `architectures` entry of the model's
# config.json (e.g. "FishOmniModel" -> the Fish Audio Drama 3 pipeline). It
# resolves that from a LOCAL DIRECTORY without touching the Hub, so all we have
# to do is hand it the directory that holds the manifest.
if [ -n "${MODEL_DOWNLOAD_FOLDER:-}" ] && { [ -f "${MODEL_DOWNLOAD_FOLDER}/config.json" ] || [ -f "${MODEL_DOWNLOAD_FOLDER}/model_index.json" ]; }; then
    echo "[startup] Operator explicitly provided valid MODEL_DOWNLOAD_FOLDER: ${MODEL_DOWNLOAD_FOLDER}"
else
    # Sort by SHALLOWEST depth first, NOT newest mtime: a component subfolder
    # (vae/, transformer/, scheduler/) carries its own config.json, and a
    # freshly-touched one would otherwise win and be loaded as the main model.
    # Mtime is only the tiebreaker for two models at the same depth on a reused
    # PVC. '|| true' stops SIGPIPE when head closes the pipe early.
    TARGET_MANIFEST=$(find "${BASE_PATH}" -maxdepth 10 -type f \( -name "config.json" -o -name "model_index.json" \) -printf '%d %T@ %p\n' 2>/dev/null | sort -k1,1n -k2,2nr | head -n 1 | cut -d' ' -f3 || true)

    if [ -z "${TARGET_MANIFEST}" ]; then
        echo "[startup] FATAL: Crawled ${BASE_PATH} (up to 10 levels) but found zero config.json / model_index.json files." >&2
        exit 1
    fi

    MODEL_DOWNLOAD_FOLDER=$(dirname "${TARGET_MANIFEST}")
    echo "[startup] Discovered active model root at: ${MODEL_DOWNLOAD_FOLDER}"
fi
export MODEL_DOWNLOAD_FOLDER

# --- 2a. Offline Codec Resolution (Drama 3 / S2-Pro) ---
# The Drama 3 pipeline's vocoder defaults its codec to a HUB repo:
#   --vocoder.factory.codec_path "fishaudio/s2-pro@1de9996b6be38b745688de084d87a5633f714e4e"
# On an air-gapped cluster that download cannot happen, and the stage dies.
# The published `fishaudio/drama-3` checkpoint ships the SAME codec.pth (same
# LFS oid 74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f),
# so when the model directory already holds one, point the vocoder at it and
# the pipeline never consults the Hub.
#
# Only applied when the file is actually present: for any other omni model the
# flag is meaningless, and for a drama-3 mirror missing codec.pth the upstream
# default is left in place (it will fail loudly on the air-gapped cluster, which
# is the correct signal -- the mirror is incomplete).
CODEC_ARGS=()
if [ -f "${MODEL_DOWNLOAD_FOLDER}/codec.pth" ]; then
    CODEC_ARGS=(--vocoder.factory.codec_path="${MODEL_DOWNLOAD_FOLDER}")
    echo "[startup] local codec.pth found -> vocoder.factory.codec_path=${MODEL_DOWNLOAD_FOLDER} (offline, no Hub lookup)"
else
    echo "[startup] no codec.pth in the model dir -> leaving vocoder.factory.codec_path at its default"
fi

# --- 3. Writable HOME + TMPDIR ---
# OICM leaves HOME=/home/runner, which is READ-ONLY. Force to /tmp.
export HOME=/tmp

# sgl-omni builds its inter-process sockets (ZMQ + msgspec handles) via
# tempfile, which honours TMPDIR. Pinning keeps every transient write on local
# ext4 instead of the NFS PVC.
export TMPDIR="${TMPDIR:-/tmp}"

# --- 4. Cache Configurations ---
# Same reasoning as the SGLang examples: keep JIT / inductor / PTX caches on
# local ephemeral disk to avoid NFS lock contention and read-only rootfs.
# These names are the ones the pinned stack actually reads (SGLang 0.5.23 +
# torch 2.14.1); sgl-omni itself has no separate cache root.
#
# ⚠️ DO NOT use "${VAR:-default}" here. That form only fires when the variable
# is UNSET, and this base image ships real values for some of them:
#     FLASHINFER_WORKSPACE_BASE=/root
# /root is correct for the base image's own root user, but OICM runs us as
# uid 10000 on a READ-ONLY rootfs, so the fallback never fires and FlashInfer's
# first JIT kernel write dies with "Permission denied" (verified: mkdir -p /root
# silently succeeds because the dir exists, so only the write fails, and it
# fails LATE — after model load, at the first kernel compile).
#
# The rule below is "defer to an inherited value ONLY if this process can
# actually write to it". That keeps operator/OCIM overrides working while
# making an unusable inherited path impossible to inherit.
_writable_dir() {
    local _var="$1" _fallback="$2" _inherited="${!1:-}"
    if [ -n "${_inherited}" ] && mkdir -p "${_inherited}" 2>/dev/null && [ -w "${_inherited}" ]; then
        printf '%s' "${_inherited}"
    else
        # NOTE: diagnostics go to STDERR. This function is consumed via $(...),
        # so anything written to stdout becomes part of the path value.
        if [ -n "${_inherited}" ]; then
            echo "[startup] ${_var}=${_inherited} is not writable -> using ${_fallback}" >&2
        fi
        mkdir -p "${_fallback}" 2>/dev/null || true
        printf '%s' "${_fallback}"
    fi
}

export SGLANG_CACHE_DIR="$(_writable_dir SGLANG_CACHE_DIR /tmp/sglang)"
export TORCHINDUCTOR_CACHE_DIR="$(_writable_dir TORCHINDUCTOR_CACHE_DIR /tmp/sglang/inductor)"
export TRITON_CACHE_DIR="$(_writable_dir TRITON_CACHE_DIR /tmp/sglang/triton)"
export CUDA_CACHE_PATH="$(_writable_dir CUDA_CACHE_PATH /tmp/nv)"
export FLASHINFER_WORKSPACE_BASE="$(_writable_dir FLASHINFER_WORKSPACE_BASE /tmp/flashinfer)"
mkdir -p "${SGLANG_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${FLASHINFER_WORKSPACE_BASE}"

# --- 5. Air-gap Enforcements ---
# Prevent HF/transformers from trying to reach the Hub and hanging at startup.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# --- 6. CUDA Forward-Compatibility ---
# CUDA the IMAGE was built for, encoded as 1000*major + 10*minor (cu130 ->
# 13000). A property of THIS image's base, not a node guess.
IMAGE_CUDA_VERSION="${IMAGE_CUDA_VERSION:-13000}"
export IMAGE_CUDA_VERSION
_cuda_probe() {
    /opt/sglang/bin/python3 - <<'PY'
import ctypes, os, sys
need = int(os.environ.get("IMAGE_CUDA_VERSION", "13000"))
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
    for COMPAT in /usr/local/cuda/compat /usr/local/cuda-13.0/compat /usr/local/cuda-13/compat; do
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
        echo "[startup] FATAL: CUDA still not usable WITH forward-compat (rc=${_probe_rc})." >&2
        exit 1
    fi
fi

# --- 7. Operator-supplied Extra Flags ---
# OICM "Model Server Arguments" -> EXTRA_ARGS (SGLANG_EXTRA_ARGS as an alias).
# arg_normalizer.py accepts every form an operator might type (=/space/quoted/
# unquoted JSON/scalar) and emits NUL-separated tokens.
#
# NOTE for sgl-omni specifically: its pipeline config is addressed with DOTTED
# LEAF flags, one value per flag, e.g.
#     --tts_engine.factory.max_running_requests 64
#     --preprocessing.factory.max_reference_audio_seconds 90
#     --vocoder.gpu 1
# Two different values for the same flag in one launch are REFUSED (not
# last-wins), and the `stages.` prefix is rejected -- the stage name is the head.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_ARGS_ARR=()
if [ -n "${EXTRA_ARGS:-}${SGLANG_EXTRA_ARGS:-}" ]; then
    while IFS= read -r -d '' _tok; do
        EXTRA_ARGS_ARR+=("${_tok}")
    done < <(cd "${_SCRIPT_DIR}" && python3 -m arg_normalizer --argv-nul)
fi

# --- 8. Launch ---
# --host 0.0.0.0 is sgl-omni's default but passed explicitly: OICM's Service
# must reach the pod, and an inherited bind-address change upstream should not
# silently make the server localhost-only.
#
# --port 8080 because OICM's Service and k8s probes target 8080; sgl-omni's own
# default is 8000.
#
# --served*-name equivalent: sgl-omni calls it --model-name and defaults it to
# the pipeline name, so MODEL_ID is only passed when the operator sets it.
NAME_ARGS=()
if [ -n "${MODEL_ID:-}" ]; then
    NAME_ARGS=(--model-name="${MODEL_ID}")
else
    echo "[startup] MODEL_ID unset -> omitting --model-name (sgl-omni defaults to the pipeline name)"
fi

LAUNCH_ARGS=(
  --host=0.0.0.0
  --port=8080
  --model-path="${MODEL_DOWNLOAD_FOLDER}"
)
LAUNCH_ARGS+=("${NAME_ARGS[@]}")
LAUNCH_ARGS+=("${CODEC_ARGS[@]}")
LAUNCH_ARGS+=("${EXTRA_ARGS_ARR[@]}")

echo "[startup] launching SGLang-Omni (extra:${EXTRA_ARGS_ARR[*]:-none})"

# --- 8a. Debug Dump (OPT-IN) ---
# Keep DEBUG_DUMP=1 for post-mortems; default OFF so boots stay fast and quiet
# (a dump fires on every restart in CrashLoopBackOff).
if [[ "${DEBUG_DUMP:-0}" == "1" ]]; then
    _debug_dump_main() {
        echo "[debug] ===== STARTUP DEBUG DUMP @ $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
        echo "[debug] identity: $(id 2>/dev/null) hostname=$(hostname) bash=${BASH_VERSION:-?}"
        echo "[debug] env (secrets redacted):"
        env 2>/dev/null | sort | while IFS='=' read -r _k _v; do
            case "${_k}" in
                *KEY*|*SECRET*|*TOKEN*|*PASSWORD*|*PASS*|*CREDENTIAL*|*ACCESS_KEY*)
                    echo "[debug]   ${_k}=<REDACTED>" ;;
                *) echo "[debug]   ${_k}=${_v}" ;;
            esac
        done
        echo "[debug] LAUNCH_ARGS=[${LAUNCH_ARGS[*]:-}] CODEC_ARGS=[${CODEC_ARGS[*]:-}]"
        echo "[debug] gpu:"; nvidia-smi 2>/dev/null | head -30 | sed 's/^/[debug]   /'
        echo "[debug] disk:"; df -h /tmp /dev/shm "${BASE_PATH:-/}" 2>/dev/null | sed 's/^/[debug]   /'
        echo "[debug] model dir:"; ls -lah "${MODEL_DOWNLOAD_FOLDER:-}" 2>/dev/null | head -25 | sed 's/^/[debug]   /'
        echo "[debug] versions:"
        /opt/sglang/bin/python3 -c 'import sglang,sglang_omni,torch; print("sglang",sglang.__version__,"| omni",getattr(sglang_omni,"__version__","?"),"| torch",torch.__version__)' 2>&1 | sed 's/^/[debug]   /'
        echo "[debug] cache writability:"
        for _d in "${SGLANG_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${FLASHINFER_WORKSPACE_BASE}" "${HOME}" "${TMPDIR}"; do
            [ -w "${_d}" ] && echo "[debug]   ${_d}: WRITABLE" || echo "[debug]   ${_d}: NOT-WRITABLE/MISSING"
        done
        echo "[debug] resolved pipeline config:"
        /opt/sglang/bin/sgl-omni config view --model-path "${MODEL_DOWNLOAD_FOLDER}" 2>&1 | head -60 | sed 's/^/[debug]   /'
        echo "[debug] ===== end dump ====="
    }
    _debug_dump_main
fi

# Resolved config, honoured by the exec below.
echo "[startup] ===== exec command ====="
printf '  %q\n' /opt/sglang/bin/sgl-omni serve "${LAUNCH_ARGS[@]}"
echo "[startup] ----- on one line -----"
echo "[startup] $ /opt/sglang/bin/sgl-omni serve ${LAUNCH_ARGS[*]}"
echo "[startup] ========================="

exec /opt/sglang/bin/sgl-omni serve "${LAUNCH_ARGS[@]}"
