#!/usr/bin/env bash
# =============================================================================
# deliver-image.sh -- ship a podman image to the Al Ain Harbor registry through
# MinIO (S3), instead of pushing from this dev box.
#
# WHY THIS EXISTS
# ---------------
# Uploading directly to the registry from here measured **2.2 MB/s**, and
# `podman push` re-uploads every layer uncompressed (`...layer.v1.tar`) even
# for layers the registry already holds as gzip -- 0% dedupe, ~39 GB, ~5 h.
#
# The relay is ~60 min for the same image because each leg runs on the network
# that suits it:
#   dev box  -> MinIO    ~8 MB/s   (mc multipart, 64 threads)
#   MinIO    -> node     ~117 MB/s (in-cluster curl, local network)
#   node     -> Harbor   27 s      (containerd push, inside the cluster)
#
# Usage:
#   ./deliver-image.sh <local-image-ref> <registry-repo-path> <tag>
#
# Example:
#   ./deliver-image.sh localhost/oicm/sglang-omni:0.2.0-cu130-amd64 \
#       openinnovationai/platform/mlops/mlops-serving/sglang-omni \
#       0.2.0-cu130-amd64
#
# Re-runnable. Every step is idempotent: the S3 object is overwritten, the
# staging pod is reused if present, and an already-imported tar is cheap.
# =============================================================================
set -euo pipefail

IMG="${1:?usage: $0 <local-image-ref> <registry-repo-path> <tag>}"
REPO="${2:?usage: $0 <local-image-ref> <registry-repo-path> <tag>}"
TAG="${3:?usage: $0 <local-image-ref> <registry-repo-path> <tag>}"

# --- knobs -------------------------------------------------------------------
KUBECONFIG_FILE="${KUBECONFIG_FILE:-$HOME/.kube/alain-oicm.conf}"
export KUBECONFIG="$KUBECONFIG_FILE"

MC_ALIAS="${MC_ALIAS:-adeo-oicm-alain}"
MC_BUCKET="${MC_BUCKET:-oi-deployment-files}"
# 64 threads is 8x faster than 8 on the Al Ain path (measured, see memory).
MC_THREADS="${MC_THREADS:-64}"

REGISTRY="${REGISTRY:-registry.adeoaiengine.ecouncil.ae}"
set +x
# Reuse the caller's inherited authorized protected environment. No default
# Basic secret is embedded in this repository, and tracing stays disabled.
REG_USER="${REG_USER:-}"
REG_PASS="${REG_PASS:-}"
export REG_USER REG_PASS
[[ -n "$REG_USER" && -n "$REG_PASS" ]] || die 'Provide existing authorized registry credentials via REG_USER and REG_PASS; no default credential is stored in this helper.'
[[ "$REG_USER" == *$'\n'* || "$REG_PASS" == *$'\n'* ]] && die 'Registry values must not contain line feeds'

STAGING_NS="${STAGING_NS:-pause-push}"
STAGING_POD="${STAGING_POD:-img-staging}"
STAGING_NODE="${STAGING_NODE:-adeo-gpu-02}"   # 525+ GB free; gpu-01 has ~155
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OBJECT="$(basename "${REPO}")-${TAG}.tar"     # e.g. sglang-omni-0.2.0-cu130-amd64.tar
TAR="/tmp/${OBJECT}"
TARGET="${REGISTRY}/${REPO}:${TAG}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- 0. preflight ------------------------------------------------------------
say "0/7  preflight"
command -v podman >/dev/null || die "podman not found"
command -v mc     >/dev/null || die "mc not found"
command -v kubectl >/dev/null || die "kubectl not found"
podman image exists "$IMG" || die "image not found locally: $IMG"
[[ -f "$KUBECONFIG" ]] || die "kubeconfig not found: $KUBECONFIG"
# grep -c, NOT grep -q: `-q` exits at the first match, the producer then dies of
# SIGPIPE, and `set -o pipefail` turns that into a spurious failure (~4% of runs).
mc --insecure alias list 2>/dev/null | grep -c "^${MC_ALIAS}" >/dev/null \
    || die "mc alias '$MC_ALIAS' not configured (run: mc alias set ...)"
kubectl get nodes >/dev/null 2>&1 || die "cannot reach cluster with $KUBECONFIG"

AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
say "      image   : $IMG"
say "      target  : $TARGET"
say "      s3 obj  : ${MC_ALIAS}/${MC_BUCKET}/${OBJECT}"
say "      free    : ${AVAIL_GB} GB on /  (tar needs ~20 GB)"

# --- 1. save -----------------------------------------------------------------
say "1/7  podman save -> $TAR"
# oci-archive (not oci-dir): a single file uploads far better via mc multipart,
# and `ctr images import` accepts it directly.
if [[ -f "$TAR" ]] && podman image exists "$IMG"; then
    SAVED_IMG=$(cat "${TAR}.src" 2>/dev/null || true)
    if [[ "$SAVED_IMG" == "$IMG" ]]; then
        say "      reusing existing $TAR (same image ref); delete it to force"
    else
        rm -f "$TAR"
    fi
fi
if [[ ! -f "$TAR" ]]; then
    podman save --format=oci-archive -o "$TAR" "$IMG"
    printf '%s' "$IMG" > "${TAR}.src"
fi
TAR_BYTES=$(stat -c %s "$TAR")
say "      $(numfmt --to=iec "$TAR_BYTES") ($TAR_BYTES bytes)"

# --- 2. upload to MinIO ------------------------------------------------------
say "2/7  upload to MinIO (${MC_THREADS} multipart threads)"
# Progress is invisible until CompleteMultipartUpload, so report elapsed time
# and the object size afterwards rather than streaming a fake percentage.
UPLOAD_START=$(date +%s)
MC_UPLOAD_MULTIPART_THREADS="$MC_THREADS" \
    mc --insecure cp --quiet "$TAR" "${MC_ALIAS}/${MC_BUCKET}/${OBJECT}"
UPLOAD_SECS=$(( $(date +%s) - UPLOAD_START ))

REMOTE_BYTES=$(mc --insecure stat --json "${MC_ALIAS}/${MC_BUCKET}/${OBJECT}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["size"])')
[[ "$REMOTE_BYTES" == "$TAR_BYTES" ]] \
    || die "size mismatch: local=$TAR_BYTES remote=$REMOTE_BYTES"
RATE=$(python3 -c "print(f'{$TAR_BYTES/1048576/max($UPLOAD_SECS,1):.1f}')")
say "      uploaded in ${UPLOAD_SECS}s (${RATE} MiB/s), size verified"

# --- 3. presigned URL --------------------------------------------------------
say "3/7  presign (24h)"
# mc share download prints THREE lines: URL:, Expire:, Share:. Only the Share
# line is signed -- grabbing the first https:// match yields a broken URL whose
# key contains both, and S3 answers SignatureDoesNotMatch (403).
mc --insecure share download --expire 24h \
    "${MC_ALIAS}/${MC_BUCKET}/${OBJECT}" 2>/dev/null \
    | sed -n 's/^Share: //p' | tr -d '\r\n' > /tmp/presign.txt
[[ -s /tmp/presign.txt ]] || die "failed to obtain presigned URL"
URL=$(cat /tmp/presign.txt)

HTTP=$(curl -sk -o /dev/null -w '%{http_code}' -r 0-1023 "$URL")
[[ "$HTTP" == "206" ]] || die "presigned URL failed validation (HTTP $HTTP, want 206)"
say "      206 Partial Content OK"

# --- 4. staging pod ----------------------------------------------------------
say "4/7  staging pod ${STAGING_NS}/${STAGING_POD} on ${STAGING_NODE}"
if ! kubectl get pod -n "$STAGING_NS" "$STAGING_POD" >/dev/null 2>&1; then
    kubectl apply -f "${SCRIPT_DIR}/img-staging-pod.yaml"
else
    say "      already exists"
fi
kubectl wait --for=condition=Ready -n "$STAGING_NS" "pod/${STAGING_POD}" --timeout=180s

# --- 5. download inside the cluster -----------------------------------------
say "5/7  download into the node (in-cluster, ~117 MB/s)"
kubectl exec -n "$STAGING_NS" "$STAGING_POD" -- bash -c "
set -euo pipefail
mkdir -p /host/tmp
if [ -f /host/tmp/${OBJECT} ] && [ \"\$(stat -c %s /host/tmp/${OBJECT})\" = '${TAR_BYTES}' ]; then
    echo '      already present on node, skipping download'
else
    curl -sk -o /host/tmp/${OBJECT} '${URL}'
    echo \"      downloaded: \$(stat -c %s /host/tmp/${OBJECT}) bytes\"
fi
[ \"\$(stat -c %s /host/tmp/${OBJECT})\" = '${TAR_BYTES}' ] || { echo 'SIZE MISMATCH' >&2; exit 1; }
"

# --- 6. import + push inside the cluster ------------------------------------
say "6/7  ctr import + tag + push"
# Al Ain containerd socket is the standard /host/run/containerd/containerd.sock
# (NOT RKE2's /run/k3s/..., which is the Abu Dhabi path).
#
# `ctr images import` names the image after the tar's own reference, i.e. the
# local podman ref (localhost/oicm/...). We therefore discover the imported ref
# rather than assuming it, retag it to the Harbor path, and push.
kubectl exec -n "$STAGING_NS" "$STAGING_POD" -- bash -c "
set -euo pipefail
CTR='/host/usr/bin/ctr -a /host/run/containerd/containerd.sock -n k8s.io'

echo '      importing (unpacks ~$(python3 -c "print(round(${TAR_BYTES}/2**30))") GiB, takes minutes)...'
\$CTR images import /host/tmp/${OBJECT} 2>&1 | tail -3

SRC=\$(\$CTR images list -q | grep -F '${IMG}' | head -1 || true)
if [ -z \"\$SRC\" ]; then
    SRC=\$(\$CTR images list -q | grep -F '${OBJECT%.tar}' | head -1 || true)
fi
if [ -z \"\$SRC\" ]; then
    echo 'ERROR: could not identify the imported image ref. Recent refs:' >&2
    \$CTR images list -q | tail -10 >&2
    exit 1
fi
echo \"      imported as: \$SRC\"

\$CTR images tag --force \"\$SRC\" '${TARGET}'
echo '      target tag ready; separate protected-auth publisher follows'
"

# Protected stdin is separate from the bash -c script and never appears in the
# outer kubectl/bash argv. ctr itself still exposes its -u argv to superusers.
printf '%s:%s\n' "$REG_USER" "$REG_PASS" | \
kubectl exec -i -n "$STAGING_NS" "$STAGING_POD" -- bash -c '
set -euo pipefail
IFS= read -r REG_BASIC
CTR=(/host/usr/bin/ctr -a /host/run/containerd/containerd.sock -n k8s.io)
TARGET="$1"
test "$("${CTR[@]}" images list -q "name==${TARGET}")" = "$TARGET"
"${CTR[@]}" images push --skip-verify -u "$REG_BASIC" "$TARGET"
' -- "$TARGET"
say '      push returned success; independent registry identity check follows'

# --- 7. verify ---------------------------------------------------------------
say "7/7  verify against the registry"
#
# Use the OCI DISTRIBUTION API (/v2/...), not Harbor's REST API
# (/api/v2.0/projects/<proj>/repositories/<repo>/artifacts/<tag>).
#
# Harbor's REST lookup returns NOT_FOUND for this repository no matter how the
# repo path is encoded, even though the repo is listed and artifact_count is
# non-zero:
#   encoded "a%252Fb"  -> NOT_FOUND
#   encoded "a%2Fb"    -> NOT_FOUND
#   with/without the "<project>/" prefix -> NOT_FOUND
# The push definitely succeeded (the manifest is served by /v2), so the REST
# endpoint is simply unreliable here. The /v2 check is also a STRONGER test: it
# compares the config digest the registry serves against the local image ID.
# Basic auth is read from protected inherited data, not command arguments.
REG_MANIFEST=$(python3 - "$REGISTRY" "$REPO" "$TAG" <<'PY'
import base64,json,os,ssl,sys,urllib.request
host,repo,tag=sys.argv[1:4]
url=f'https://{host}/v2/{repo}/manifests/{tag}'
headers={
 'Accept':'application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json',
 'Authorization':'Basic '+base64.b64encode(f"{os.environ['REG_USER']}:{os.environ['REG_PASS']}".encode()).decode(),
}
try:
 with urllib.request.urlopen(urllib.request.Request(url,headers=headers),context=ssl._create_unverified_context(),timeout=90) as response:
  body=response.read()
  assert response.status == 200
  manifest=json.loads(body)
  assert manifest.get('config',{}).get('digest'), 'Body is not an image manifest'
  print(body.decode(),end='')
except Exception as exc:
 raise SystemExit('Registry readback failed: '+type(exc).__name__) from None
PY
)

[[ -n "$REG_MANIFEST" ]] || die "registry returned an empty manifest for ${TARGET}"

REG_CONFIG=$(printf '%s' "$REG_MANIFEST" | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["config"]["digest"])')
# `podman image inspect --format '{{.Id}}'` yields BARE hex (no "sha256:"),
# while the registry manifest reports a prefixed digest. Normalise both to the
# prefixed form before comparing, otherwise identical images compare unequal.
LOCAL_CONFIG=$(podman image inspect --format '{{.Id}}' "$IMG")
case "$LOCAL_CONFIG" in
    sha256:*) ;;
    *) LOCAL_CONFIG="sha256:${LOCAL_CONFIG}" ;;
esac

# .Id IS the config digest, so this is a byte-identity check on the image
# config (env, labels, entrypoint, rootfs layer list) -- not just "a tag exists".
if [[ "$REG_CONFIG" != "$LOCAL_CONFIG" ]]; then
    printf '\n  local config : %s\n  registry     : %s\n' \
        "$LOCAL_CONFIG" "$REG_CONFIG" >&2
    die "digest mismatch: the registry does NOT hold the image we built"
fi

printf '%s' "$REG_MANIFEST" | python3 -c '
import json, sys
d = json.load(sys.stdin)
layers = d.get("layers", [])
total = sum(l.get("size", 0) for l in layers)
print("  config digest :", d["config"]["digest"])
print("  mediaType     :", d.get("mediaType"))
print("  layers        : %d  (%.2f GiB compressed)" % (len(layers), total / 2**30))
' | tee /tmp/verify-out.txt

MANIFEST_DIGEST=$(sed -n 's/^[Dd]ocker-[Cc]ontent-[Dd]igest: *//p' \
    /tmp/verify-headers.txt | tr -d '\r\n')
say "      manifest digest: ${MANIFEST_DIGEST:-<not returned>}"
say "      config digest matches the local build -- image is correctly in Harbor"

rm -f /tmp/verify-headers.txt /tmp/verify-out.txt

say "DONE -- ${TARGET}"
cat <<EOF

Next:
  # rollout so the deployment picks up the new image
  KUBECONFIG=$KUBECONFIG kubectl -n adeo rollout restart deployment/<name>
  KUBECONFIG=$KUBECONFIG kubectl -n adeo rollout status  deployment/<name>

  # clean up staging when finished
  kubectl -n ${STAGING_NS} delete pod ${STAGING_POD}
EOF
