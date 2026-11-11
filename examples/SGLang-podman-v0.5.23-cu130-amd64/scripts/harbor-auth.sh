#!/usr/bin/env bash
# Registry authentication controller: existing standard protected store only.
# Consumes credsStore/credentialHelpers through Podman instead of raw decoding.
set -euo pipefail
set +x
REGISTRY="${REGISTRY:-registry.adeoaiengine.ecouncil.ae}"
MODE="${1:?Usage: harbor-auth.sh preflight [<local-image-ref>] | push-only <local-image-ref>}"
SOURCE_IMAGE="${2:-}"
[[ "$MODE" == preflight || "$MODE" == push-only ]] || {
    echo 'Usage: harbor-auth.sh preflight [<local-image-ref>] | push-only <local-image-ref>' >&2
    exit 64
}
case "$SOURCE_IMAGE" in
    *:*) SOURCE_TAG="${SOURCE_IMAGE##*:}" ;;
    '') SOURCE_TAG=0.5.23-cu130-amd64 ;;
    *) echo 'Source image must carry a tag' >&2; exit 64 ;;
esac
TARGET="$REGISTRY/openinnovationai/platform/mlops/mlops-serving/sglang:$SOURCE_TAG"
AUTH_FILE="${REGISTRY_AUTH_FILE:-$HOME/.config/containers/auth.json}"
if [[ ! -f "$AUTH_FILE" ]]; then
    printf 'Configure Podman auth in standard auth.json or set REGISTRY_AUTH_FILE. No manual secrets introduced in history.\n' >&2
    exit 2
fi
STORE_TYPE="$(python3 - "$AUTH_FILE" "$REGISTRY" <<'PY'
import json,sys
from pathlib import Path
p,reg=sys.argv[1:]
if Path(p).is_symlink() or Path(p).stat().st_mode & 0o077:
    raise SystemExit('Registry auth file must be a regular mode-0600 file')
d=json.load(open(p))
helpers=d.get('credentialHelpers') or {}
if d.get('credStore') or d.get('credsStore') or helpers.get(reg):
    print('DELEGATED')
elif (d.get('auths') or {}).get(reg,{}).get('auth'):
    print('INLINE')
else:
    print('MISSING')
PY
)"
[[ "$STORE_TYPE" == DELEGATED || "$STORE_TYPE" == INLINE ]] || {
    echo 'No exact-registry configured authentication or unsupported file' >&2
    exit 2
}
if [[ "$STORE_TYPE" == DELEGATED ]]; then
    podman --authfile "$AUTH_FILE" --tls-verify=false manifest inspect \
      "$TARGET" >/dev/null
    echo 'PASS: delegated existing Podman registry auth read (no secret written or decoded)'
    if [[ "$MODE" == preflight ]]; then
        echo 'PASS: exact existing registry auth; no uploads or writes'
        exit 0
    fi
elif [[ "$MODE" == preflight ]]; then
    python3 - "$AUTH_FILE" "$REGISTRY" "$TARGET" <<'PY'
import hashlib, json, ssl, sys
from pathlib import Path
from urllib.request import Request, urlopen
p,host,target=sys.argv[1:]
d=json.loads(Path(p).read_text())
basic=(d.get('auths') or {}).get(host,{}).get('auth')
if not basic:
    raise SystemExit('Exact inline registry auth missing')
repo,_,tag=target.removeprefix(host+'/').rpartition(':')
request=Request(f'https://{host}/v2/{repo}/manifests/{tag}', headers={
    'Accept':'application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json',
    'Authorization':'Basic '+basic,
})
try:
    with urlopen(request,context=ssl._create_unverified_context(),timeout=60) as r:
        body=r.read()
        h=r.headers.get('Docker-Content-Digest','')
        if r.status!=200 or not h.startswith('sha256:') or hashlib.sha256(body).hexdigest()!=h[7:]:
            raise SystemExit('Expected valid digest-authenticated manifest response')
        m=json.loads(body)
        if not m.get('config') or not m.get('layers'):
            raise SystemExit('Registry body is not an image manifest')
        print('PASS: exact registry auth and manifest digest readback; no writes or secret output')
except Exception as exc:
    raise SystemExit('Registry readback failed: '+type(exc).__name__) from None
PY
else
    echo 'Existing inline registry auth found; Podman will reuse its protected file'
fi
case "$MODE" in
    preflight)
        echo 'PASS: exact existing registry auth; no uploads or writes'
        ;;
    push-only)
        [[ "${DIRECT_PUSH_CONFIRMED_ALLOW_SLOW:-}" == required ]] || {
            echo 'Direct workstation push is measured slow for initial large archives. Do not select this for by-default HF delivery; use Make target deliver-sglang-podman-0.5.23-cu130.' >&2
            exit 64
        }
        [[ -n "$SOURCE_IMAGE" ]] || { echo 'Required source-image-ref positional argument' >&2; exit 64; }
        podman image exists "$SOURCE_IMAGE"
        echo 'Protected Podman auth preflight; explicit operator has chosen direct measured-slow route.'
        podman tag "$SOURCE_IMAGE" "$TARGET"
        podman push --authfile "$AUTH_FILE" --tls-verify=false "$TARGET"
        podman --authfile "$AUTH_FILE" --tls-verify=false manifest inspect "$TARGET" >/dev/null
        echo 'Direct push command and registry readback completed; compare returned config digest with source config separately.'
        ;;
    *)
        echo 'Usage: preflight <image> or push-only <image>' >&2; exit 64
        ;;
esac
