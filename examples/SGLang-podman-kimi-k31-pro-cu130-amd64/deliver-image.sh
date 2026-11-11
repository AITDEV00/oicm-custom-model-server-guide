#!/usr/bin/env bash
# =============================================================================
# deliver-image.sh -- ship a podman image to the Al Ain Harbor registry through
# MinIO (S3), instead of pushing from this dev box.
#
# WHY THIS EXISTS
# ---------------
# Uploading directly to the registry from here measured **2.2 MB/s**, and
# `podman push` re-uploads every layer uncompressed (`...layer.v1.tar`) even for
# layers the registry already holds as gzip -- 0% dedupe, ~5 h for a 40 GB
# image. The relay is faster because each leg runs on the network that suits it:
#
#   dev box  -> MinIO    ~8 MB/s   (mc multipart, 64 threads)
#   MinIO    -> node     ~117 MB/s (in-cluster curl, local network)
#   node     -> Harbor   27 s      (containerd push, inside the cluster)
#
# HOW RE-DELIVERY STAYS FAST  (the point of this image's Dockerfile ordering)
# ---------------------------------------------------------------------------
# Kimi-K3.1-Pro is 40 GB on disk / ~15.4 GiB compressed, but every byte except
# the OICM harness comes from the pinned upstream base. The Dockerfile keeps
# startup.sh + arg_normalizer.py as the LAST single COPY, so editing them
# rebuilds in ~6 s and changes only 2 layers out of 80 (measured).
#
# This script extends that to the wire. It works on an OCI *directory* rather
# than a monolithic tar, keeps that directory on the node between runs, and
# transfers only blobs the node does not already have:
#
#   1. `podman save --format=oci-dir` -- one file per blob, so a re-save after a
#      harness edit rewrites only the changed blob(s). An oci-archive rewrites
#      all ~16 GB every time.
#   2. A JSON ledger (`_delivery.json`) in S3 records which blob digests were
#      delivered under this tag. It is written LAST, so an aborted delivery
#      never claims blobs it did not finish uploading.
#   3. `mc mirror` the oci-dir up; it skips objects whose size matches, so a
#      re-delivery normally moves a few KB.
#   4. On the node, fetch only the missing blobs, tar the oci-dir locally (disk
#      speed, no network) and `ctr images import`.
#
# Measured shape: first delivery is dominated by the ~16 GiB upload; a
# harness-only re-delivery is dominated by the node's local tar+import because
# the wire transfer is ~14 KiB.
#
# `FRESH=1` ignores the ledger and re-uploads every blob.
#
# Usage:
#   ./deliver-image.sh <local-image-ref> <registry-repo-path> <tag>
#
# Example:
#   ./deliver-image.sh localhost/oicm/sglang:kimi-k31-pro-cu130-amd64 \
#       openinnovationai/platform/mlops/mlops-serving/sglang \
#       kimi-k31-pro-cu130-amd64
#
# Re-runnable and idempotent at every step.
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
# Reuse the caller's inherited authorized protected environment without logging
# the values. No default Basic password is embedded in this repository.
REG_USER="${REG_USER:-}"
REG_PASS="${REG_PASS:-}"
export REG_USER REG_PASS
[[ "$REG_USER" == *$'\n'* || "$REG_PASS" == *$'\n'* ]] && die 'Registry values must not contain line feeds'
[[ -n "$REG_USER" && -n "$REG_PASS" ]] || die 'Provide existing authorized registry credentials via REG_USER and REG_PASS; no default credential is stored in this helper.'
REG_USER="$REG_USER" REG_PASS="$REG_PASS" python3 - "$REGISTRY" "$REPO" "$TAG" --manifest-read <<'PY'
import base64,os,ssl,sys,urllib.request
host,repo,tag=sys.argv[1:4]
auth=base64.b64encode(f"{os.environ['REG_USER']}:{os.environ['REG_PASS']}".encode()).decode()
try:
 request=urllib.request.Request(f'https://{host}/v2/{repo}/manifests/{tag}',headers={'Accept':'application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json','Authorization':'Basic '+auth})
 with urllib.request.urlopen(request,context=ssl._create_unverified_context(),timeout=90) as r:
  assert r.status == 200 and r.read(1)
  print('Registry authentication readback available')
except Exception as exc:
 raise SystemExit('Registry preflight failed: '+type(exc).__name__) from None
PY

STAGING_NS="${STAGING_NS:-pause-push}"
STAGING_POD="${STAGING_POD:-img-staging}"
STAGING_NODE="${STAGING_NODE:-adeo-gpu-02}"   # 525+ GB free; gpu-01 has ~155
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# FRESH=1 -> ignore the ledger and re-upload every blob.
FRESH="${FRESH:-0}"

NAME="$(basename "${REPO}")-${TAG}"           # sglang-kimi-k31-pro-cu130-amd64
OCIDIR="/tmp/${NAME}.oci"
LEDGER="/tmp/${NAME}.delivery.json"
URLMAP="/tmp/${NAME}.urls.txt"
REMOTE_PREFIX="${NAME}.oci"                   # S3 "directory" holding the oci-dir
TARGET="${REGISTRY}/${REPO}:${TAG}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
kb()  { python3 -c "import sys;n=float(sys.argv[1]);print(f'{n/1024:.1f} KiB' if n<2**20 else f'{n/2**20:.2f} MiB')" "$1"; }

# --- 0. preflight ------------------------------------------------------------
say "0/8  preflight"
for c in podman mc kubectl python3 tar; do command -v "$c" >/dev/null || die "$c not found"; done
podman image exists "$IMG" || die "image not found locally: $IMG"
[[ -f "$KUBECONFIG" ]] || die "kubeconfig not found: $KUBECONFIG"
# grep -c, NOT grep -q: `-q` exits at the first match, the producer then dies of
# SIGPIPE, and `set -o pipefail` turns that into a spurious failure (~4% of runs,
# which is how this check once rejected a perfectly good mc config).
mc --insecure alias list 2>/dev/null | grep -c "^${MC_ALIAS}" >/dev/null \
    || die "mc alias '$MC_ALIAS' not configured (run: mc alias set ...)"
kubectl get nodes >/dev/null 2>&1 || die "cannot reach cluster with $KUBECONFIG"

AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
LOCAL_CONFIG="sha256:$(podman image inspect --format '{{.Id}}' "$IMG" | sed 's/^sha256://')"
say "      image   : $IMG"
say "      config  : $LOCAL_CONFIG"
say "      target  : $TARGET"
say "      free    : ${AVAIL_GB} GB on /"

# --- 1. save as an OCI directory --------------------------------------------
say "1/8  podman save --format=oci-dir -> $OCIDIR"
mkdir -p "$OCIDIR"
podman save --format=oci-dir -o "$OCIDIR" "$IMG" >/dev/null
[[ -f "$OCIDIR/index.json" ]] || die "oci-dir has no index.json"

read -r BLOB_COUNT BLOB_BYTES < <(python3 - "$OCIDIR" <<'PY'
import json, os, sys
d = sys.argv[1]
seen = set()
idx = json.load(open(os.path.join(d, "index.json")))
for m in idx.get("manifests", []):
    seen.add(m["digest"])
    mf = os.path.join(d, "blobs", "sha256", m["digest"].split(":")[1])
    if not os.path.exists(mf):
        continue
    man = json.load(open(mf))
    c = man.get("config", {}).get("digest")
    if c:
        seen.add(c)
    for l in man.get("layers", []):
        seen.add(l["digest"])
total = sum(os.path.getsize(os.path.join(d, "blobs", "sha256", x.split(":")[1]))
            for x in seen
            if os.path.exists(os.path.join(d, "blobs", "sha256", x.split(":")[1])))
print(len(seen), total)
PY
)
say "      $(du -sh "$OCIDIR" | cut -f1) on disk, $BLOB_COUNT blobs, $(kb "$BLOB_BYTES") of blob data"

# --- 2. what has already been delivered? ------------------------------------
say "2/8  read the delivery ledger from MinIO"
: > /tmp/delivered.txt
if mc --insecure stat "${MC_ALIAS}/${MC_BUCKET}/${REMOTE_PREFIX}/_delivery.json" >/dev/null 2>&1; then
    mc --insecure cat "${MC_ALIAS}/${MC_BUCKET}/${REMOTE_PREFIX}/_delivery.json" \
        > "$LEDGER" 2>/dev/null || true
    if [[ -s "$LEDGER" ]]; then
        python3 - "$LEDGER" > /tmp/delivered.txt <<'PY'
import json, sys
for d in json.load(open(sys.argv[1])).get("blobs", []):
    print(d)
PY
    fi
    say "      ledger found: $(wc -l < /tmp/delivered.txt) blob(s) already delivered"
else
    say "      no ledger: first delivery (everything goes up)"
fi

if [[ "$FRESH" == "1" ]]; then
    say "      FRESH=1 -> ignoring the ledger"
    : > /tmp/delivered.txt
fi

# --- 3. upload the oci-dir --------------------------------------------------
say "3/8  mirror the oci-dir to MinIO (${MC_THREADS} threads)"
UPLOAD_START=$(date +%s)
# Blobs are CONTENT-ADDRESSED: same digest == same bytes, always. So a size
# comparison is a complete correctness test, and we must NOT pass --overwrite.
# --overwrite compares mtimes, and every `podman save` writes fresh mtimes, so
# it re-uploads all 15.36 GiB on every run even when every blob is already in
# S3 (measured: a no-op re-delivery took 991s instead of 0s).
# With FRESH=1 we DO want to re-send, hence the conditional.
if [[ "$FRESH" == "1" ]]; then
    MIRROR_FLAGS=(--overwrite)
else
    MIRROR_FLAGS=()
fi
# Without --overwrite, mc still compares mtime and prints one `mm-source-mtime`
# ERROR line per object whose local copy looks newer. That is EXPECTED and
# harmless here (blobs are content-addressed, so an existing object with the
# right digest already has the right bytes). Filter those lines so they do not
# look like a failure; anything else on stderr still comes through.
# A temp file is used instead of `2> >(grep ...)`: process substitution leaves an
# async child behind, and a later heredoc/stdin redirect in the same script can
# then attach to the wrong stream (this actually broke step 6 once).
MC_UPLOAD_MULTIPART_THREADS="$MC_THREADS" \
    mc --insecure mirror --quiet "${MIRROR_FLAGS[@]}" \
        "$OCIDIR" "${MC_ALIAS}/${MC_BUCKET}/${REMOTE_PREFIX}" \
        2> /tmp/mc-mirror.err || true
grep -v 'mm-source-mtime' /tmp/mc-mirror.err >&2 || true
rm -f /tmp/mc-mirror.err
UPLOAD_SECS=$(( $(date +%s) - UPLOAD_START ))
say "      mirrored in ${UPLOAD_SECS}s"

REMOTE_BYTES=$(mc --insecure stat --json "${MC_ALIAS}/${MC_BUCKET}/${REMOTE_PREFIX}/index.json" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["size"])')
LOCAL_BYTES=$(stat -c %s "$OCIDIR/index.json")
[[ "$REMOTE_BYTES" == "$LOCAL_BYTES" ]] \
    || die "index.json size mismatch (local=$LOCAL_BYTES remote=$REMOTE_BYTES)"
say "      index.json verified in S3 ($LOCAL_BYTES bytes)"

# --- 4. build the fetch list ------------------------------------------------
# Only objects the node is missing get a presigned URL. On a re-delivery that is
# the index, the two tiny bookkeeping files and (usually) one 4 KB layer.
say "4/8  presign the objects the node needs"
S3PREFIX="${MC_ALIAS}/${MC_BUCKET}/${REMOTE_PREFIX}"
python3 - "$OCIDIR" "$S3PREFIX" /tmp/delivered.txt > "$URLMAP" <<'PY'
import json, os, subprocess, sys
ocidir, s3prefix, delivered_file = sys.argv[1], sys.argv[2], sys.argv[3]
have = set()
if os.path.exists(delivered_file):
    have = {l.strip() for l in open(delivered_file) if l.strip()}

need = ["index.json", "oci-layout"]
idx = json.load(open(os.path.join(ocidir, "index.json")))
for m in idx.get("manifests", []):
    need.append("blobs/sha256/" + m["digest"].split(":")[1])
    mf = os.path.join(ocidir, "blobs", "sha256", m["digest"].split(":")[1])
    if not os.path.exists(mf):
        continue
    man = json.load(open(mf))
    c = man.get("config", {}).get("digest")
    if c:
        need.append("blobs/sha256/" + c.split(":")[1])
    for l in man.get("layers", []):
        need.append("blobs/sha256/" + l["digest"].split(":")[1])

seen = set()
for rel in need:
    if rel in seen:
        continue
    seen.add(rel)
    digest = "sha256:" + os.path.basename(rel) if rel.startswith("blobs/") else ""
    if not os.path.exists(os.path.join(ocidir, rel)):
        print("%s\tmissing\t" % rel)
        continue
    # Always mint a URL, including for `skip` entries. The ledger records what
    # was SENT to the node, but the node's scratch dir is a hostPath that can be
    # deleted independently of the ledger -- so a skip must be re-verifiable on
    # the node and, if the blob is gone, fetched. Presigning is a local HMAC, it
    # costs nothing and transfers nothing.
    out = subprocess.run(["mc", "--insecure", "share", "download", "--expire", "24h",
                          "%s/%s" % (s3prefix, rel)],
                         capture_output=True, text=True).stdout
    url = ""
    for line in out.splitlines():
        if line.startswith("Share: "):
            url = line[len("Share: "):].strip()
    action = "skip" if (digest and digest in have) else "fetch"
    print("%s\t%s\t%s" % (rel, action, url))
PY

TO_FETCH=$(awk -F'\t' '$2=="fetch"' "$URLMAP" | wc -l)
TO_SKIP=$(awk -F'\t' '$2=="skip"' "$URLMAP" | wc -l)
FETCH_BYTES=$(awk -F'\t' '$2=="fetch"{print $1}' "$URLMAP" | while read -r rel; do
    stat -c %s "$OCIDIR/$rel" 2>/dev/null || echo 0
done | awk '{s+=$1} END {printf "%d", s+0}')
awk -F'\t' '$2=="missing"{print "      MISSING locally: "$1}' "$URLMAP"
say "      $TO_FETCH object(s) to send ($(kb "$FETCH_BYTES")), $TO_SKIP already on the node"

FIRST_URL=$(awk -F'\t' '$2=="fetch"{print $3; exit}' "$URLMAP")
if [[ -n "$FIRST_URL" ]]; then
    HTTP=$(curl -sk -o /dev/null -w '%{http_code}' -r 0-1023 "$FIRST_URL")
    [[ "$HTTP" == "206" ]] || die "presigned URL failed validation (HTTP $HTTP, want 206)"
    say "      206 Partial Content OK"
else
    say "      nothing to transfer -- the node already holds this exact image"
fi

# --- 5. staging pod ----------------------------------------------------------
say "5/8  staging pod ${STAGING_NS}/${STAGING_POD} on ${STAGING_NODE}"
if ! kubectl get pod -n "$STAGING_NS" "$STAGING_POD" >/dev/null 2>&1; then
    kubectl apply -f "${SCRIPT_DIR}/img-staging-pod.yaml"
else
    say "      already exists"
fi
kubectl wait --for=condition=Ready -n "$STAGING_NS" "pod/${STAGING_POD}" --timeout=180s

# --- 6. fetch only the missing blobs ---------------------------------------
say "6/8  fetch into the node (in-cluster, ~117 MB/s)"
# NOTE: do NOT use `kubectl cp` here. It shells out to `tar` INSIDE the container
# to build the stream, and the nettest image ships no tar (it also ships no
# python3). stdin redirection needs nothing beyond `cat`, which it does have.
kubectl exec -i -n "$STAGING_NS" "$STAGING_POD" -- bash -c 'cat > /host/tmp/urls.txt' < "$URLMAP"
kubectl exec -n "$STAGING_NS" "$STAGING_POD" -- bash -c "
set -euo pipefail
D=/host/tmp/${NAME}.oci
mkdir -p \"\$D/blobs/sha256\"
fetched=0; reused=0; healed=0
while IFS=\$'\t' read -r rel action url; do
    [[ -z \"\$rel\" ]] && continue
    case \"\$action\" in
      skip)
        # The ledger says this was delivered, but the ledger is not the node:
        # /host/tmp is a hostPath that can be wiped independently. Verify on the
        # node and fall back to fetching -- a stale ledger must never produce an
        # INCOMPLETE oci-dir.
        if [ -s \"\$D/\$rel\" ]; then
            reused=\$((reused+1))
        else
            mkdir -p \"\$(dirname \"\$D/\$rel\")\"
            curl -sk -o \"\$D/\$rel\" \"\$url\"
            healed=\$((healed+1))
        fi ;;
      fetch)
        mkdir -p \"\$(dirname \"\$D/\$rel\")\"
        curl -sk -o \"\$D/\$rel\" \"\$url\"
        fetched=\$((fetched+1)) ;;
      missing)
        echo \"      ERROR: \$rel missing locally; cannot assemble a complete oci-dir\" >&2
        exit 1 ;;
    esac
done < /host/tmp/urls.txt
echo \"      fetched=\$fetched reused=\$reused healed=\$healed\"
# The container has no python3; the HOST does, and chroot /host gives it the
# host's own libs. (`/host/usr/bin/python3` run directly fails: it cannot find
# libexpat.so.1 -- the container's lib path is not the host's.)
# NOTE: inside the chroot /host becomes /, so strip the prefix from the path.
chroot /host /usr/bin/python3 - \"\${D#/host}\" <<'PY2'
import json, os, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, 'index.json')))
missing = []
for m in idx.get('manifests', []):
    mf = os.path.join(d, 'blobs', 'sha256', m['digest'].split(':')[1])
    if not os.path.exists(mf):
        missing.append(m['digest']); continue
    man = json.load(open(mf))
    c = man.get('config', {}).get('digest')
    if c and not os.path.exists(os.path.join(d, 'blobs', 'sha256', c.split(':')[1])):
        missing.append(c)
    for l in man.get('layers', []):
        if not os.path.exists(os.path.join(d, 'blobs', 'sha256', l['digest'].split(':')[1])):
            missing.append(l['digest'])
if missing:
    sys.exit('INCOMPLETE oci-dir on node: %d blob(s) missing, e.g. %s' % (len(missing), missing[:3]))
print('      oci-dir complete')
PY2
"

# --- 7. tar locally, import, tag, push --------------------------------------
say "7/8  tar on the node, ctr import, tag, push"
# `ctr images import` needs a tar (an oci-archive is just an oci-layout dir in a
# tar), not a directory. Tarring on the node is local disk I/O and means a
# monolithic archive never crosses the network.
#
# Al Ain's containerd socket is /host/run/containerd/containerd.sock
# (NOT RKE2's /run/k3s/..., which is the Abu Dhabi path).
kubectl exec -n "$STAGING_NS" "$STAGING_POD" -- bash -c "
set -euo pipefail
CTR='/host/usr/bin/ctr -a /host/run/containerd/containerd.sock -n k8s.io'
D=/host/tmp/${NAME}.oci
T=/host/tmp/${NAME}.tar

echo '      tarring the oci-dir on the node (local disk, no network)...'
rm -f \"\$T\"
# Host binaries: the container has neither tar nor du, but both run fine from
# /host (they only need libc, which the Fedora base provides). MUST cd into the
# oci-dir first: `ctr import` expects an oci-archive whose root is ./blobs,
# ./index.json, ./oci-layout -- an absolute-path tar would not satisfy that.
cd \"\$D\" && /host/usr/bin/tar -cf \"\$T\" .
echo \"      tar: \$(/host/usr/bin/du -h \"\$T\" | cut -f1)\"

echo '      ctr images import...'
\$CTR images import \"\$T\" 2>&1 | tail -3

SRC=\$(\$CTR images list -q | grep -F '${IMG}' | head -1 || true)
if [ -z \"\$SRC\" ]; then
    SRC=\$(\$CTR images list -q | grep -F '${NAME}' | head -1 || true)
fi
if [ -z \"\$SRC\" ]; then
    echo 'ERROR: could not identify the imported image ref. Recent refs:' >&2
    \$CTR images list -q | tail -10 >&2
    exit 1
fi
echo \"      imported as: \$SRC\"
\$CTR images tag --force "\$SRC" '${TARGET}'
echo '      target tag ready; separate protected-auth publisher follows'
# Free only the TRANSIENT tar. Keep the oci-dir: it is the node-side blob cache
# that makes a re-delivery skip 77 of 79 objects, and blobs are
# content-addressed so a stale entry can never be wrong. /host/tmp is a hostPath
# on the node's real disk, so it is NOT reclaimed when the pod is deleted.
rm -f "\$T"
echo '      node tar removed (oci-dir kept as the blob cache)'
"

# Preserve registry credentials through protected exec stdin, not arguments to
# kubectl or its enclosing bash. The eventual ctr child still has a literal -u
# argument visible to privileged local users; this does not hide that limitation.
printf '%s:%s\n' "$REG_USER" "$REG_PASS" | \
kubectl exec -i -n "$STAGING_NS" "$STAGING_POD" -- bash -c '
set -euo pipefail
IFS= read -r REG_BASIC
CTR=(/host/usr/bin/ctr -a /host/run/containerd/containerd.sock -n k8s.io)
TARGET="$1"
test "$("${CTR[@]}" images list -q "name==${TARGET}")" = "$TARGET"
"${CTR[@]}" images push --skip-verify -u "$REG_BASIC" "$TARGET"
' -- "$TARGET"

# The ledger is written only after the push succeeded, so a future run can
# safely skip these blobs.
say "8/8  record the delivery and verify against the registry"
DELIVER_IMG_REF="$IMG" python3 - "$OCIDIR" "$LOCAL_CONFIG" "$TAG" "$S3PREFIX" > "$LEDGER" <<'PY'
import json, os, sys
d, cfg, tag, s3 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
idx = json.load(open(os.path.join(d, "index.json")))
blobs = set()
for m in idx.get("manifests", []):
    blobs.add(m["digest"])
    mf = os.path.join(d, "blobs", "sha256", m["digest"].split(":")[1])
    if not os.path.exists(mf):
        continue
    man = json.load(open(mf))
    if man.get("config", {}).get("digest"):
        blobs.add(man["config"]["digest"])
    for l in man.get("layers", []):
        blobs.add(l["digest"])
print(json.dumps({
    "tag": tag,
    "config": cfg,
    "image_ref": os.environ.get("DELIVER_IMG_REF", ""),
    "s3_prefix": s3,
    "blobs": sorted(blobs),
}, indent=2))
PY
mc --insecure cp --quiet "$LEDGER" "${S3PREFIX}/_delivery.json"

# OCI DISTRIBUTION API, not Harbor's REST API: the REST lookup returns
# NOT_FOUND for this repository however the repo path is encoded, even though
# the repo is listed and artifact_count is non-zero. The /v2 check is also a
# STRONGER test -- it compares the config digest the registry serves against the
# local image ID, proving the bytes in Harbor are the bytes we built. Basic auth
# is read from inherited protected operator input, never put in command argv.
REG_MANIFEST=$(python3 - "$REGISTRY" "$REPO" "$TAG" <<'PY'
import base64,json,os,ssl,sys,urllib.request
host,repo,tag=sys.argv[1:4]
user=os.environ['REG_USER']
url=f'https://{host}/v2/{repo}/manifests/{tag}'
headers={
 'Accept':'application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json',
 'Authorization':'Basic '+base64.b64encode(f"{user}:{os.environ['REG_PASS']}".encode()).decode(),
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
'

MANIFEST_DIGEST=$(sed -n 's/^[Dd]ocker-[Cc]ontent-[Dd]igest: *//p' \
    /tmp/verify-headers.txt | tr -d '\r\n')
say "      manifest digest: ${MANIFEST_DIGEST:-<not returned>}"
say "      config digest matches the local build -- image is correctly in Harbor"
rm -f /tmp/verify-headers.txt

say "DONE -- ${TARGET}"
cat <<EOF

Next:
  # rollout so the deployment picks up the new image
  KUBECONFIG=$KUBECONFIG kubectl -n adeo rollout restart deployment/<name>
  KUBECONFIG=$KUBECONFIG kubectl -n adeo rollout status  deployment/<name>

  # The staging pod is left running on purpose: it holds /host/tmp/${NAME}.oci,
  # the node-side blob cache that makes a re-delivery skip 77 of 79 objects.
  # Delete it (and/or that directory) only if you need the node's disk back.
  kubectl -n ${STAGING_NS} delete pod ${STAGING_POD}
EOF
