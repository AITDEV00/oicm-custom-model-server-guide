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
#     mixed: H200 nodes run R580 / CUDA 13.0 and support the cu129 image NATIVELY,
#     while older nodes may be on R570 / CUDA 12.8 and need the forward-compat libs
#     the base image ships at /usr/local/cuda/compat. We probe the native driver
#     and prepend compat ONLY when the host actually needs it -- forcing compat on
#     a driver that is already new enough loads an OLDER user-mode libcuda than the
#     kernel module and triggers CUDA error 803 at init. See the compat section.
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

# --- Dynamic Model Discovery (Dirty PVC & Subfolder Safe) ---
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

# CUDA's PTX-JIT cache defaults to ~/.nv (=/home/runner/.nv, READ-ONLY). The JIT
# can fire (forward-compat path, or any PTX-only kernel), so give it a writable dir.
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/tmp/nv}"

# Pre-create the two dirs we introduced (the OICM /tmp ones already exist).
mkdir -p "${VLLM_CACHE_ROOT}" "${CUDA_CACHE_PATH}"

# --- Air-gap: never reach for the HuggingFace Hub (egress is blocked) ---

# Without these, HF/transformers may try the Hub and hang on the network
# timeout at startup. The model is already on disk, so offline is correct.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# --- CUDA forward-compatibility: enable ONLY when the host driver needs it ----
#
# The base image ships CUDA forward-compat user-mode driver libs (an R575-era
# libcuda + ptxjitcompiler + nvvm) under /usr/local/cuda/compat. Forward-compat
# is a ONE-WAY bridge: it lets a NEWER CUDA userspace run on an OLDER kernel
# driver. It is valid ONLY when the host kernel driver is OLDER than the image's
# CUDA. If the host driver already supports the image's CUDA (same or newer --
# e.g. an H200 on R580 / CUDA 13.0 running this cu129 image), prepending the
# compat libcuda DOWNGRADES the user-mode driver below the kernel module, and
# the driver rejects the pair with CUDA error 803 (cudaErrorSystemDriverMismatch)
# at cuInit -> the engine never starts -> CrashLoopBackOff.
#
# The fleet is mixed and node drivers get upgraded under us, so we DECIDE at
# runtime instead of hardcoding a driver version. The decision mirrors CUDA's
# OWN sufficiency rule: the host is fine on its native libcuda iff the driver's
# max supported CUDA (cuDriverGetVersion) is >= the CUDA the image was built for.
#   * native sufficient -> use host libcuda, do NOT add compat.
#   * native too old     -> prepend compat, then re-probe and fail loud if still bad.
# Why the version check and not just "does cuInit succeed": on a genuinely old
# node (e.g. R570 / CUDA 12.8) cuInit on the native libcuda SUCCEEDS, but vLLM's
# bundled CUDA 12.9 runtime would then fail later with cudaErrorInsufficientDriver
# -- exactly the case compat is for. Checking the driver version up front catches
# that, while the cuInit/device check still catches the 803 mismatch when compat
# is wrongly in play. Correct for old, current, and future drivers, nothing to
# maintain except the image's own CUDA constant below (tied to the base image).

# CUDA the IMAGE was built for, encoded as 1000*major + 10*minor (cu129 -> 12090).
# This is a property of THIS image's base (FROM ...cu129...), not a node guess;
# bump it only when the base image's CUDA changes.
IMAGE_CUDA_VERSION="${IMAGE_CUDA_VERSION:-12090}"

# Fast CUDA probe via ctypes (no torch import -> sub-second). Uses the driver API
# (libcuda.so.1 -- stable SONAME across CUDA majors). Exit codes:
#   0   native/current libcuda is SUFFICIENT (cuInit OK, driver CUDA >= image, >=1 dev)
#   10  driver too old: cuDriverGetVersion < image CUDA  -> needs compat
#   100 init OK but no device visible
#   201 no loadable libcuda at all
#   <CUDA rc> cuInit/cuDeviceGetCount returned a nonzero CUDA error (e.g. 803)
# Reads IMAGE_CUDA_VERSION from the environment; honours whatever LD_LIBRARY_PATH
# is in effect when called.
export IMAGE_CUDA_VERSION
_cuda_probe() {
    python3 - <<'PY'
import ctypes, os, sys
need = int(os.environ.get("IMAGE_CUDA_VERSION", "12090"))
try:
    lib = ctypes.CDLL("libcuda.so.1")
except OSError:
    sys.exit(201)                       # no loadable libcuda at all
# cuDriverGetVersion does NOT require cuInit and reports the MAX CUDA the
# currently-loaded driver supports (e.g. 13000 for R580, 12080 for R570).
drv = ctypes.c_int(0)
if lib.cuDriverGetVersion(ctypes.byref(drv)) != 0:
    sys.exit(204)
sys.stderr.write("[probe] libcuda max CUDA=%d, image needs=%d\n" % (drv.value, need))
rc = lib.cuInit(0)
if rc != 0:
    sys.exit(rc if rc < 256 else 202)   # nonzero CUDA error, e.g. 803
if drv.value < need:
    sys.exit(10)                        # driver too old for the image's runtime
n = ctypes.c_int()
rc = lib.cuDeviceGetCount(ctypes.byref(n))
if rc != 0:
    sys.exit(rc if rc < 256 else 203)
sys.exit(0 if n.value > 0 else 100)     # 100 = init OK but no device visible
PY
}

# Informational only (the probe makes the decision, not this string). nvidia-smi
# resolves the CUDA version from whichever libcuda is on the path, so read it
# BEFORE we touch LD_LIBRARY_PATH to reflect the true native host driver.
_HOST_CUDA="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | head -1 || true)"

if _cuda_probe; then
    echo "[startup] native host driver is sufficient for the image's CUDA (${_HOST_CUDA:-version unknown}); forward-compat NOT needed -> using host libcuda directly."
else
    _probe_rc=$?
    # rc=100 means the driver was fine but no GPU was visible -- compat won't fix
    # that (it's an allocation/scheduling problem), so don't mask it.
    if [ "${_probe_rc}" -eq 100 ]; then
        echo "[startup] FATAL: CUDA initialized but no GPU is visible to this pod (rc=100); check the accelerator allocation, not forward-compat." >&2
        exit 1
    fi
    echo "[startup] native libcuda not sufficient (rc=${_probe_rc}; 10=driver older than image CUDA); host driver=${_HOST_CUDA:-version unknown} -> attempting forward-compat."
    _compat_applied=0
    # Tolerant of a path change in a future base-image bump.
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
    # Re-probe WITH compat so a genuine mismatch fails here with a clear message
    # instead of deep inside vLLM with an opaque traceback.
    if _cuda_probe; then
        echo "[startup] forward-compat CUDA check OK -> proceeding."
    else
        _probe_rc=$?
        echo "[startup] FATAL: CUDA still not usable WITH forward-compat (rc=${_probe_rc}); driver/runtime mismatch is beyond what these compat libs can bridge." >&2
        exit 1
    fi
fi

# --- Operator-supplied extra flags (OICM "Model Server Arguments" -> EXTRA_ARGS) ---
#
# OICM passes the field VERBATIM into EXTRA_ARGS (inner double quotes intact --
# diagnostic-confirmed by reading the env on a live pod). The ONLY transform
# applied to it is the normalizer below.
#
# /app/arg_normalizer.py accepts EVERY form an operator might type, so nobody
# has to remember a quoting rule (this replaces the old "MUST single-quote JSON"
# constraint, which was a footgun):
#   --speculative-config={"method":"mtp","num_speculative_tokens":2}   (unquoted = JSON)
#   --speculative-config='{"method":"mtp","num_speculative_tokens":2}' (single-quoted)
#   --speculative-config '{"method":"mtp","num_speculative_tokens":2}' (space form)
#   --spec-method=mtp --spec-tokens=2                                   (scalar)
#   --speculative-config.method=mtp ...                                (dot-notation)
# It does three things, all in Python stdlib (shlex/json -- no pip install):
#   1. PROTECT unquoted JSON: before shlex.split would strip the inner " (turning
#      {"method":"mtp"} into the invalid {method:mtp}), wrap any value starting
#      with { or [ in shlex.quote so it round-trips intact. Detection is
#      STRUCTURAL (value shape), so it works for every current JSON flag AND
#      every future one without any hardcoded list to maintain.
#   2. FOLD space form (--flag value) into --flag=value; keep booleans bare.
#   3. DEDUP repeated flags last-wins (argparse-style) -- absorbs an accidental
#      double-paste that vLLM would otherwise reject as 'Found duplicate keys'.
# We still NEVER use `eval` (brace-expansion + injection risk); tokens come back
# over a NUL delimiter so bash can never re-split them. The normalizer logs any
# auto-fixes/dropped flags/invalid-JSON to stderr (visible in `kubectl logs`).
#
# This is the SAME module the diagnostic image's POST /test-args endpoint uses,
# so anything that passes /test-args will work here, and anything that fails
# /test-args fails here -- one source of truth.
# Locate arg_normalizer.py next to this script so the entrypoint works no
# matter where OICM runs it from (and doesn't depend on a hardcoded /app).
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_ARGS_ARR=()
if [ -n "${EXTRA_ARGS:-}${VLLM_EXTRA_ARGS:-}" ]; then
    while IFS= read -r -d '' _tok; do
        EXTRA_ARGS_ARR+=("${_tok}")
    done < <(cd "${_SCRIPT_DIR}" && python3 -m arg_normalizer --argv-nul)
fi

# --- Tensor parallelism: match the allocation, but never break MIG ----------

# OICM injects NUM_GPUS for the allocation (the diagnostic showed NUM_GPUS=1 on
# a MIG slice). TP should equal the GPU count on a multi-GPU node, but MUST stay
# 1 on a MIG slice -- a MIG instance is a single isolated partition with no
# NVLink/P2P, so --tensor-parallel-size > 1 fails there. We detect MIG from the
# MIG- prefix in NVIDIA_VISIBLE_DEVICES. Skip entirely if the operator already
# set any parallelism flag -- we now scan the NORMALIZED tokens (covers =,
# space, and short forms uniformly) instead of substring-matching the raw
# string (which missed cases like `--tensor-parallel-size=2` with no trailing
# space).
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
    # MIG slice -> force single-GPU regardless of NUM_GPUS.
    if [[ "${NVIDIA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
        TP_ARGS=(--tensor-parallel-size=1)
        echo "[startup] MIG slice detected -> tensor-parallel-size=1"
    # Multiple full GPUs -> shard across them (only if NUM_GPUS is a number > 1).
    elif [[ "${NUM_GPUS:-1}" =~ ^[0-9]+$ ]] && [ "${NUM_GPUS:-1}" -gt 1 ]; then
        TP_ARGS=(--tensor-parallel-size="${NUM_GPUS}")
        echo "[startup] ${NUM_GPUS} GPUs -> tensor-parallel-size=${NUM_GPUS}"
        # TP>1 uses /dev/shm for inter-worker IPC; a small shm causes Bus errors.
        echo "[startup] NOTE: TP>1 needs adequate /dev/shm; raise it in OICM if you see 'Bus error'."
    fi
    # else: single full GPU -> leave vLLM's default of 1 (no flag needed).
else
    echo "[startup] operator set a parallelism flag -> leaving TP to operator args"
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