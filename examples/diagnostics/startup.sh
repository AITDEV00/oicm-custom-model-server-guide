#!/bin/bash
# OICM environment diagnostic. Never aborts early: every probe reports
# pass/fail rather than crashing, so you get the full picture in one run.
# Output goes to stdout (kubectl logs) AND, if a writable dir exists, to a
# file that the HTTP server on :8080 re-serves for easy retrieval.

LINE="==============================================================="

# Capture how THIS script was invoked (the k8s command/args land here as $@).
SCRIPT_ARGV=("$@")
SCRIPT_ARGC="$#"

# ---- pick a writable spot for the report file (best effort) ----
REPORT_FILE=""
for d in /tmp /tmp/oicm /dev/shm /home/runner /app; do
    if mkdir -p "$d" 2>/dev/null && touch "$d/.diag_w" 2>/dev/null; then
        rm -f "$d/.diag_w" 2>/dev/null
        REPORT_FILE="$d/oicm-diag-report.txt"
        break
    fi
done
export REPORT_FILE

probe_write() {
    local p="$1"
    if mkdir -p "$p" 2>/dev/null && touch "$p/.diag_w" 2>/dev/null; then
        rm -f "$p/.diag_w" 2>/dev/null
        printf '  WRITABLE   %s\n' "$p"
    elif [ -d "$p" ]; then
        printf '  RO/DENIED  %s (exists, not writable)\n' "$p"
    else
        printf '  MISSING    %s (cannot create)\n' "$p"
    fi
}

net_test() {
    local url="$1"
    local code
    code=$(curl -sS -m 5 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null)
    if [ -n "$code" ] && [ "$code" != "000" ]; then
        printf '  REACHABLE  %-40s (HTTP %s)\n' "$url" "$code"
    else
        printf '  BLOCKED    %-40s (no response / timeout)\n' "$url"
    fi
}

# returns 0 (true) if version $1 >= version $2
ver_ge() {
    [ "$1" = "$2" ] && return 0
    local smaller
    smaller=$(printf '%s\n%s\n' "$1" "$2" | sort -V 2>/dev/null | head -n1)
    [ "$smaller" = "$2" ]
}

run_report() {
echo "$LINE"
echo "          OICM RUNTIME DIAGNOSTIC REPORT"
echo "          generated: $(date -u 2>/dev/null) UTC"
echo "$LINE"

echo
echo "## 1. IDENTITY & SECURITY CONTEXT"
echo "  id            : $(id 2>/dev/null || echo 'id failed')"
echo "  whoami        : $(whoami 2>/dev/null || echo 'no passwd entry for UID (expected under arbitrary-UID)')"
echo "  effective UID : $(id -u 2>/dev/null)"
echo "  effective GID : $(id -g 2>/dev/null)"
echo "  groups        : $(id -G 2>/dev/null)"
echo "  HOME          : ${HOME}"
echo "  expanduser ~  : $(python3 -c 'import os;print(os.path.expanduser("~"))' 2>/dev/null)"
echo "  PWD           : $(pwd)"
echo "  capabilities  : $(grep -E 'CapEff' /proc/self/status 2>/dev/null | awk '{print $2}')"
echo "  SA token mount: $([ -d /var/run/secrets/kubernetes.io/serviceaccount ] && echo 'PRESENT (k8s SA mounted)' || echo 'absent')"

echo
echo "## 2. ROOT FILESYSTEM MODE"
rootopts=$(awk '$2=="/"{print $4}' /proc/mounts 2>/dev/null | head -1)
echo "  / mount opts  : ${rootopts:-unknown}"
case "$rootopts" in
    ro,*|*,ro|*,ro,*) echo "  => root filesystem is READ-ONLY (readOnlyRootFilesystem: true)";;
    rw,*|*,rw|*,rw,*) echo "  => root filesystem is writable";;
    *) echo "  => could not determine; see write probes below";;
esac

echo
echo "## 3. WRITE-PROBE: standard paths"
for p in / /home /home/runner /app /tmp /tmp/oicm /dev/shm /var/tmp; do
    probe_write "$p"
done

echo
echo "## 4. WRITE-PROBE: likely model/volume mounts"
for p in "${PVC_PATH:-}" /pvc-home /data-volume /data /models /model /workspace /vllm-workspace /mnt; do
    [ -z "$p" ] && continue
    probe_write "$p"
done

echo
echo "## 5. WRITE-PROBE: cache dirs the real vLLM image needs"
BASE_GUESS="${PVC_PATH:-/tmp/oicm}"
for p in "${BASE_GUESS}/cache" "${HOME}/.cache" "${HOME}/.triton" "/tmp/torchinductor_test" "${HOME}/.nv"; do
    probe_write "$p"
done

echo
echo "## 6. MOUNTS (rw/ro flags)"
grep -vE '^(proc|sysfs|cgroup|tmpfs /sys|devpts|mqueue) ' /proc/mounts 2>/dev/null \
  | awk '{printf "  %-28s %-10s %s\n", $2, $3, $4}' | head -40

echo
echo "## 7. DISK SPACE"
df -h 2>/dev/null | awk 'NR==1 || /pvc|data|tmp|shm|overlay|model/ {print "  "$0}'

echo
echo "## 8. MODEL PRESENCE CHECK"
MDL="${MODEL_DOWNLOAD_FOLDER:-${BASE_GUESS}/app/download/base_model}"
echo "  expected model dir: $MDL"
if [ -d "$MDL" ]; then
    echo "  dir exists. contents (top level):"
    ls -la "$MDL" 2>/dev/null | head -25 | sed 's/^/    /'
    echo "  config.json    : $([ -f "$MDL/config.json" ] && echo present || echo MISSING)"
    echo "  *.safetensors  : $(ls "$MDL"/*.safetensors 2>/dev/null | wc -l) file(s)"
    echo "  approx size    : $(du -sh "$MDL" 2>/dev/null | awk '{print $1}')"
else
    echo "  => model dir NOT present (init container may run separately, or wrong path)"
fi

echo
echo "## 9. GPU / DRIVER / CUDA 12.9 (cu129) SUPPORT"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "  --- full nvidia-smi ---"
    nvidia-smi 2>&1 | sed 's/^/  /'
    echo "  --- parsed ---"
    DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | paste -sd, - 2>/dev/null)
    # "CUDA Version" in the nvidia-smi header = highest CUDA the driver supports
    MAXCUDA=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
    echo "  GPU                 : ${NAME:-unknown}"
    echo "  Driver version      : ${DRV:-unknown}"
    echo "  Max CUDA (driver)   : ${MAXCUDA:-unknown}  (highest CUDA this driver supports natively)"
    echo "  Compute capability  : ${CC:-unknown}"
    echo "  This image built for: CUDA 12.9 (cu129)"
    echo "  Reference minimums  : native 12.9 = R575 series (>=575.51.03); 12.x minor-compat floor = 525.60.13"

    if [ -n "$MAXCUDA" ]; then
        if ver_ge "$MAXCUDA" "12.9"; then
            echo "  VERDICT             : OK - driver natively supports CUDA >= 12.9. cu129 fully supported."
        elif ver_ge "$MAXCUDA" "12.0" && { [ -z "$DRV" ] || ver_ge "$DRV" "525.60.13"; }; then
            echo "  VERDICT             : MARGINAL - driver's max CUDA is ${MAXCUDA} (< 12.9)."
            echo "                        cu129 MAY run via CUDA 12.x minor-version forward compatibility,"
            echo "                        but NVIDIA restricts features in that mode - notably PTX JIT compile."
            echo "                        vLLM + Triton + torch.compile JIT-compile kernels constantly, so this"
            echo "                        is risky. Recommend upgrading the node driver to R575 for native 12.9."
        else
            echo "  VERDICT             : NOT SUPPORTED - driver too old for CUDA 12.9."
            echo "                        Need R575 (>=575.51.03) for native, or >=525.60.13 for limited compat."
        fi
    else
        echo "  VERDICT             : could not parse driver CUDA version; inspect full nvidia-smi above."
    fi

    case "$CC" in
        *9.0*|*8.9*|*10.*|*12.*) echo "  FP8 KV-cache (Qwen3.6): compute cap supports FP8 (Ada/Hopper/Blackwell). Good." ;;
        "" ) echo "  FP8 KV-cache (Qwen3.6): compute_cap unavailable on this driver; check GPU model above." ;;
        *) echo "  FP8 KV-cache (Qwen3.6): compute cap ${CC} may NOT support FP8 (needs >= 8.9). Verify." ;;
    esac
else
    echo "  nvidia-smi unavailable or no GPU visible to this pod."
    echo "  => Either no GPU was requested for this diagnostic pod, or the NVIDIA"
    echo "     container runtime isn't injecting the driver. Deploy with the same"
    echo "     GPU request as a real model to get a meaningful reading."
    ls -l /dev/nvidia* 2>/dev/null | sed 's/^/  /' || echo "  no /dev/nvidia* devices present"
fi
echo "  CUDA_VISIBLE_DEVICES   = ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  NVIDIA_VISIBLE_DEVICES = ${NVIDIA_VISIBLE_DEVICES:-<unset>}"

echo
echo "## 10. RESOURCE LIMITS (cgroup)"
memmax=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)
echo "  memory limit  : ${memmax:-unknown} bytes"
echo "  nproc         : $(nproc 2>/dev/null)"
echo "  /dev/shm size : $(df -h /dev/shm 2>/dev/null | awk 'NR==2{print $2" (avail "$4")"}')"
echo "  (note: small /dev/shm breaks tensor-parallel; vLLM needs --ipc=host or a big shm volume)"

echo
echo "## 11. NETWORK / AIR-GAP CHECK"
echo "  DNS for huggingface.co: $(getent hosts huggingface.co 2>/dev/null | awk '{print $1}' | head -1 || echo 'no resolution')"
net_test "https://huggingface.co"
net_test "https://github.com"
net_test "https://pypi.org"
# Internal endpoints if OICM injected them as env vars:
for v in MINIO_ENDPOINT S3_ENDPOINT AWS_ENDPOINT_URL HARBOR_URL OBJECT_STORE_URL; do
    val="${!v:-}"
    [ -n "$val" ] && net_test "$val"
done

echo
echo "## 12. ENVIRONMENT VARIABLES (secrets redacted)"
echo "  (look here for what OICM injects: INIT_CONTAINER_USE_PVC, PVC_PATH,"
echo "   USE_DATA_VOLUME, MODEL_ID, MODEL_DOWNLOAD_FOLDER, and any extras)"
while IFS= read -r kv; do
    k="${kv%%=*}"; v="${kv#*=}"
    case "$k" in
        *TOKEN*|*SECRET*|*PASSWORD*|*PASSWD*|*KEY*|*CRED*|*ACCESS*)
            echo "  ${k}=<redacted len=${#v}>" ;;
        *) echo "  ${k}=${v}" ;;
    esac
done < <(env | sort)

echo
echo "## 13. TOOLING PRESENT"
for t in python3 curl wget nvidia-smi git gcc ss ip; do
    printf '  %-12s %s\n' "$t" "$(command -v $t 2>/dev/null || echo 'absent')"
done

echo
echo "## 14. HOW THIS CONTAINER WAS LAUNCHED (command + args)"
echo "  Args passed to this entrypoint (== the k8s 'args:' / docker CMD): count=${SCRIPT_ARGC}"
if [ "${SCRIPT_ARGC}" -gt 0 ]; then
    i=1
    for a in "${SCRIPT_ARGV[@]}"; do echo "    arg[$i]=${a}"; i=$((i+1)); done
    echo "  => OICM DOES pass args through. You can use them for extra vLLM flags."
else
    echo "    (none)"
    echo "  => No args reached the entrypoint. Either OICM does not set k8s 'args:',"
    echo "     or it overrides 'command:' entirely. Use the VLLM_EXTRA_ARGS env hook,"
    echo "     and confirm with the pod spec in section 15 below."
fi
echo "  PID 1 cmdline    : $(tr '\0' ' ' < /proc/1/cmdline 2>/dev/null)"
echo "  entrypoint cmdline: $(tr '\0' ' ' < /proc/$$/cmdline 2>/dev/null)"
echo "  ENTRYPOINT/CMD baked in image is /app/startup.sh; anything beyond it is from k8s."

echo
echo "## 15. POD SPEC FROM K8S API (best effort - reveals exact env/args OICM set)"
KSA=/var/run/secrets/kubernetes.io/serviceaccount
if [ -r "$KSA/token" ] && [ -n "${KUBERNETES_SERVICE_HOST:-}" ]; then
    NS=$(cat "$KSA/namespace" 2>/dev/null)
    POD="${POD_NAME:-${HOSTNAME:-$(hostname 2>/dev/null)}}"
    APISERVER="https://${KUBERNETES_SERVICE_HOST}:${KUBERNETES_SERVICE_PORT_HTTPS:-443}"
    echo "  namespace=${NS}  pod=${POD}"
    echo "  querying ${APISERVER}/api/v1/namespaces/${NS}/pods/${POD} ..."
    RESP=$(curl -sS -m 8 \
        --cacert "$KSA/ca.crt" \
        -H "Authorization: Bearer $(cat "$KSA/token")" \
        -w $'\n__HTTP__%{http_code}' \
        "${APISERVER}/api/v1/namespaces/${NS}/pods/${POD}" 2>/dev/null)
    CODE="${RESP##*__HTTP__}"
    BODY="${RESP%$'\n'__HTTP__*}"
    echo "  HTTP ${CODE}"
    if [ "$CODE" = "200" ]; then
        echo "$BODY" | python3 - <<'PYEOF'
import sys, json
def red(name):
    n=name.upper()
    return any(s in n for s in ("TOKEN","SECRET","PASSWORD","PASSWD","KEY","CRED","ACCESS"))
try:
    pod=json.load(sys.stdin)
except Exception as e:
    print("    (could not parse pod JSON: %s)"%e); sys.exit(0)
spec=pod.get("spec",{})
psc=spec.get("securityContext",{})
print("    pod.securityContext: runAsUser=%s runAsNonRoot=%s fsGroup=%s fsGroupChangePolicy=%s"%(
    psc.get("runAsUser"),psc.get("runAsNonRoot"),psc.get("fsGroup"),psc.get("fsGroupChangePolicy")))
print("    volumes:")
for v in spec.get("volumes",[]):
    kind=[k for k in v.keys() if k!="name"]
    print("      - %s (%s)"%(v.get("name"),",".join(kind)))
def dump_containers(label, lst):
    for c in lst or []:
        print("    [%s] %s  image=%s"%(label,c.get("name"),c.get("image")))
        print("      command: %s"%c.get("command"))
        print("      args   : %s"%c.get("args"))
        sc=c.get("securityContext",{}) or {}
        print("      securityContext: runAsUser=%s runAsNonRoot=%s readOnlyRootFilesystem=%s allowPrivilegeEscalation=%s"%(
            sc.get("runAsUser"),sc.get("runAsNonRoot"),sc.get("readOnlyRootFilesystem"),sc.get("allowPrivilegeEscalation")))
        res=c.get("resources",{}) or {}
        print("      resources: limits=%s requests=%s"%(res.get("limits"),res.get("requests")))
        print("      volumeMounts:")
        for m in c.get("volumeMounts",[]) or []:
            print("        - %s -> %s (readOnly=%s)"%(m.get("name"),m.get("mountPath"),m.get("readOnly",False)))
        print("      env:")
        for e in c.get("env",[]) or []:
            if "value" in e:
                val= "<redacted>" if red(e.get("name","")) else e.get("value")
                print("        - %s=%s"%(e.get("name"),val))
            elif "valueFrom" in e:
                vf=e["valueFrom"]; src=list(vf.keys())[0] if vf else "?"
                print("        - %s <- %s (reference)"%(e.get("name"),src))
        froms=c.get("envFrom",[]) or []
        if froms:
            print("      envFrom: %s"%[list(f.keys()) for f in froms])
dump_containers("init", spec.get("initContainers"))
dump_containers("main", spec.get("containers"))
PYEOF
    elif [ "$CODE" = "403" ]; then
        echo "    Forbidden: this pod's ServiceAccount lacks 'get pods' RBAC."
        echo "    (Common under strict tenant isolation. The env dump in section 12"
        echo "     and the args in section 14 are then your source of truth.)"
    else
        echo "    Could not retrieve pod spec. Body (truncated):"
        echo "$BODY" | head -c 400 | sed 's/^/      /'
    fi
else
    echo "  No usable ServiceAccount token / API host -> cannot query pod spec."
    echo "  Rely on section 12 (env) and section 14 (args) instead."
fi

echo
echo "$LINE"
echo "  END OF REPORT"
echo "$LINE"
}

# Emit to stdout (logs) and to the report file if we found a writable one.
if [ -n "$REPORT_FILE" ]; then
    run_report 2>&1 | tee "$REPORT_FILE"
    echo "[diag] report also written to: $REPORT_FILE"
else
    run_report 2>&1
    echo "[diag] no writable dir found for report file; logs only"
fi

echo "[diag] starting keep-alive HTTP server on :8080"
echo "[diag] it will log which path the cluster health-check probes (see '[probe]' lines)"
exec python3 /app/serve_report.py