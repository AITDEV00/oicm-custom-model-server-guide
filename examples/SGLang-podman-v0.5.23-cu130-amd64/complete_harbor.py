#!/usr/bin/env python3
"""Import the exact SGLang v0.5.23 transport archive and publish its tag.

Reviewed continuation code, not a claim that the completed 2026-11-11 delivery
was repeated. An initial registry verification returns before any node writes.
The presigned URL and Basic credential are sent only over kubectl stdin, never
as local or remote bash positional arguments. ctr 1.7.28 still requires its
literal ``-u user:password`` child argv while pushing: stdin does NOT hide that
argument from privileged process inspection. No authfile is claimed or created.
No source archive, staging pod, or model deployment is removed or modified.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import ssl
import stat
import subprocess
import urllib.error
import urllib.parse
import urllib.request

TARGET = 'registry.adeoaiengine.ecouncil.ae/openinnovationai/platform/mlops/mlops-serving/sglang:0.5.23-cu130-amd64'
HOST, repo_tag = TARGET.split('/', 1)
REPO, TAG = repo_tag.rsplit(':', 1)
EXPECTED_CONFIG = 'sha256:ab8fc368a60473e329bde89157d84b20ce3bef8fddd5b6bf4999df65c4392cf5'
EXPECTED_MANIFEST = 'sha256:01e0d5637028f59e2836bfc8c385c393a9d722d4b971acbcfa66875569eee960'
SOURCE_SHA256 = '570648111409bdece6ec93192643808a30787311059c90f451e487cfe145c20a'
SIZE = 39_787_218_432
READY_FILES = 5
READY_BYTES = 39_787_221_225
READY_BASELINE = '2026-11-11T07:03:00Z'
CONTENT_BYTES = 39_786_978_626
REVISION = '25bb4121e30d40d2257ed7b9007763f7ca0c3b0e'
HFS3_REPO = 'maxJasper/sglang-0.5.23-cu130-amd64'
PREFIX = 'runtime/sglang-0.5.23-cu130-amd64-hf'
REMOTE_TAR = '/host/tmp/sglang-0.5.23-cu130-amd64.tar'



def _redact(value, secrets):
    if value is None:
        return ''
    if isinstance(value, bytes):
        value = value.decode(errors='replace')
    for secret in secrets:
        value = value.replace(secret, '<protected>')
    return value


def kube(*args: str, input_text: str | None = None, sensitive_input=False, timeout=None):
    """Exec a staged argv, optionally streaming data; never a credential argv.

    ``pass_fds=3`` is unnecessary: ``-i`` and subprocess ``input`` connect the
    protected data to fd 0. The bash program remains in ``-c`` argv, so stdin is
    available for exactly one URL/credential read (or the Python archive reader).
    """
    env = dict(os.environ)
    env['KUBECONFIG'] = env.get('KUBECONFIG_FILE', str(Path.home() / '.kube/alain-oicm.conf'))
    if args[:1] == ('get',):
        # Staging preflight is a Kubernetes API read, not an exec of `get` in
        # the container. Protected exec streams use the separate branch below.
        command = ['kubectl', *args]
    else:
        command = ['kubectl', 'exec', '-n', 'pause-push', 'img-staging', '--quiet']
        if input_text is not None:
            command.append('-i')
        command += ['--', *args]
    secrets = []
    if sensitive_input and input_text:
        secrets = [input_text, input_text.rstrip('\r\n')]
        # Remote tool failures may echo either the joined value or just the
        # password. Redact both; never include Basic credentials/URLs in logs.
        if '\n' not in input_text.rstrip('\r\n'):
            value = input_text.rstrip('\r\n')
            if '://' in value:
                secrets.append(urllib.parse.urlsplit(value).query)
            elif ':' in value:
                secrets.append(value.partition(':')[2])
    try:
        result = subprocess.run(command, input=input_text, text=True, capture_output=True,
                                env=env, timeout=timeout, check=True)
    except subprocess.CalledProcessError as error:
        stdout = _redact(error.stdout, secrets).strip()
        stderr = _redact(error.stderr, secrets).strip()
        detail = '; '.join(f'{name}: {value}' for name, value in
                           (('stdout', stdout), ('stderr', stderr)) if value)
        raise RuntimeError(f'Staged command exited {error.returncode}; {detail or "output was empty"}') from None
    except subprocess.TimeoutExpired as error:
        stdout = _redact(error.stdout, secrets).strip()
        stderr = _redact(error.stderr, secrets).strip()
        detail = '; '.join(f'{name}: {value}' for name, value in
                           (('stdout', stdout), ('stderr', stderr)) if value)
        raise RuntimeError(f'Staged command timed out; {detail or "output was empty"}') from None
    result.stdout = _redact(result.stdout, secrets)
    result.stderr = _redact(result.stderr, secrets)
    return result


def _credential_file(path: Path, seen: set[Path]) -> tuple[str, str]:
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise SystemExit(f'Credential store must have mode exactly 0600: {path}')
    identity = path.resolve()
    if identity in seen:
        raise SystemExit('Credential authfile pointer contains a cycle')
    seen.add(identity)
    text = path.read_text()
    if text.startswith('authfile:'):
        raw_nested = text.removeprefix('authfile:').strip()
        nested = Path(raw_nested.removeprefix('authfile:')).expanduser()
        if not nested.is_absolute():
            nested = path.parent / nested
        if not nested.is_file():
            raise SystemExit(f'Configured credential file does not exist: {nested}')
        return _credential_file(nested, seen)
    stripped = text.strip()
    if stripped.startswith('{'):
        try:
            document = json.loads(stripped)
        except ValueError:
            raise SystemExit(f'Credential store is not valid JSON: {path}') from None
        if document.get('credStore') or document.get('credsStore') or document.get('credentialHelpers'):
            raise SystemExit('Delegated registry credential helpers are not decoded; configure an existing protected inline basic-auth store for this helper.')
        entry = (document.get('auths') or {}).get(HOST)
        if not isinstance(entry, dict) or not entry.get('auth'):
            raise SystemExit(f'No inline basic auth for exact registry {HOST} in {path}')
        try:
            text = base64.b64decode(entry['auth'], validate=True).decode()
        except (ValueError, UnicodeDecodeError, binascii.Error):
            raise SystemExit(f'Invalid inline basic auth for exact registry in {path}') from None
    user, separator, password = text.rstrip('\r\n').partition(':')
    if not separator or not user or not password:
        raise SystemExit(f'Protected registry file {path} needs user:password for {HOST}')
    return user, password


def existing_credentials() -> tuple[str, str]:
    """Use an explicitly configured 0600 store, or standard container stores."""
    supplied = os.environ.get('REGISTRY_CREDS_FILE')
    paths = [Path(supplied.removeprefix('authfile:')).expanduser()] if supplied else [
        Path('/run/user') / str(os.getuid()) / 'containers/auth.json',
        Path.home() / '.config/containers/auth.json',
        Path.home() / '.docker/config.json',
    ]
    for path in paths:
        if path.is_file():
            return _credential_file(path, set())
    if supplied:
        raise SystemExit(f'Configured credential file does not exist: {paths[0]}')
    raise SystemExit('No standard existing basic-auth store is configured. Set REGISTRY_CREDS_FILE to an established protected user:password or auth.json file; never supply a credential in argv.')


def registry(path, creds, tls):
    user, password = creds
    headers = {'Authorization': 'Basic ' + base64.b64encode(f'{user}:{password}'.encode()).decode(),
               'Accept': 'application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json'}
    request = urllib.request.Request(f'https://{HOST}/v2/{REPO}/{path}', headers=headers)
    try:
        return urllib.request.urlopen(request, context=tls, timeout=90)
    except urllib.error.HTTPError as error:
        return error


def _stream_blob(response, expected_digest, expected_size):
    if response.status != 200:
        raise RuntimeError(f'Registry blob read returned HTTP {response.status}; body suppressed')
    digest = hashlib.sha256()
    count = 0
    with response:
        while chunk := response.read(16 * 1024 * 1024):
            digest.update(chunk)
            count += len(chunk)
    if count != expected_size or digest.hexdigest() != expected_digest.removeprefix('sha256:'):
        raise RuntimeError('Registry blob did not match its ordered manifest descriptor')


def verify_registry(creds, tls, check_blobs=False):
    response = registry('manifests/' + TAG, creds, tls)
    code = getattr(response, 'status', None) or response.code
    if code == 404:
        print('INFO: exact registry manifest does not exist yet', flush=True)
        return False
    if code != 200:
        raise RuntimeError(f'Registry manifest read HTTP {code}; response body suppressed')
    with response:
        body = response.read()
        digest = response.headers.get('Docker-Content-Digest')
    try:
        manifest = json.loads(body)
    except ValueError:
        raise RuntimeError('Registry returned non-JSON manifest') from None
    if digest != EXPECTED_MANIFEST or hashlib.sha256(body).hexdigest() != EXPECTED_MANIFEST.removeprefix('sha256:'):
        raise RuntimeError('Registry raw manifest digest does not match the expected source manifest')
    if manifest.get('mediaType') != 'application/vnd.docker.distribution.manifest.v2+json':
        raise RuntimeError('Registry media type differs from the live verified Docker schema 2 manifest')
    config = manifest.get('config') or {}
    layers = manifest.get('layers')
    if config.get('digest') != EXPECTED_CONFIG or not isinstance(layers, list) or len(layers) != 75:
        raise RuntimeError('Registry source config or 75-layer membership does not match')
    # Hashing the exact body above also pins descriptor order, not merely count.
    for item in layers:
        if not isinstance(item.get('size'), int) or item['size'] <= 0:
            raise RuntimeError('Registry manifest has an invalid layer size')
        item_digest = item.get('digest', '')
        if not item_digest.startswith('sha256:') or len(item_digest) != 71:
            raise RuntimeError('Registry manifest has an invalid layer digest')
        if check_blobs:
            _stream_blob(registry('blobs/' + item_digest, creds, tls), item_digest, item['size'])
    if check_blobs:
        if isinstance(config.get('size'), int) and config['size'] > 0:
            _stream_blob(registry('blobs/' + EXPECTED_CONFIG, creds, tls), EXPECTED_CONFIG, config['size'])
        else:
            raise RuntimeError('Registry manifest has no valid config size')
    print('PASS: exact registry tag, raw manifest digest, source config and 75 ordered layers', flush=True)
    print('PASS: optional full config/layer blob verification:', check_blobs, flush=True)
    print(TARGET, flush=True)
    return True


def check_minio(alias):
    mc = os.environ.get('MC', 'mc')
    raw = subprocess.check_output([mc, '--insecure', 'cat', f'{alias}/oi-deployment-files/{PREFIX}/_READY.json'], text=True)
    ready = json.loads(raw)
    expected = {'status': 'READY', 'repo': HFS3_REPO, 'revision': REVISION,
                'bucket': 'oi-deployment-files', 'prefix': PREFIX,
                'files': READY_FILES, 'bytes': READY_BYTES, 'verified': 'hash'}
    if {key: ready.get(key) for key in expected} != expected:
        raise RuntimeError('MinIO READY aggregate does not match five files and 39,787,221,225 bytes at the exact revision')
    # This aggregate includes transport metadata. Only guard against a marker
    # predating the observed baseline; do not require equality with that time.
    try:
        completed = datetime.fromisoformat(ready['completed_at'].replace('Z', '+00:00'))
        baseline = datetime.fromisoformat(READY_BASELINE.replace('Z', '+00:00'))
    except (KeyError, TypeError, ValueError):
        raise RuntimeError('MinIO READY marker has no valid completed_at timestamp') from None
    if completed.tzinfo is None or completed.utcoffset() is None:
        raise RuntimeError('MinIO READY timestamp must include an explicit UTC offset')
    if completed.astimezone(timezone.utc) < baseline:
        raise RuntimeError('MinIO READY marker predates the reviewed completion baseline')
    raw = subprocess.check_output([mc, '--insecure', 'stat', '--json',
        f'{alias}/oi-deployment-files/{PREFIX}/sglang-0.5.23-cu130-amd64.tar'], text=True)
    info = json.loads(raw)
    if info.get('size') != SIZE or info.get('etag') != '16114660a316db9969f58a1da9bf04fd-2372':
        raise RuntimeError('MinIO archive size or committed multipart identity differs')
    print('PASS: exact revision and five-file READY aggregate; tar size/committed parts match', flush=True)
    print('Multipart ETag is not SHA-256; raw node hashing and archive metadata remain required.', flush=True)


def verify_node_tar():
    checker = r'''set -euo pipefail
A="$1"; EXPECTED="$2"; BYTES="$3"
command -v sha256sum >/dev/null
[[ "$(stat -c%s "$A")" == "$BYTES" ]]
printf '%s  %s\n' "$EXPECTED" "$A" | sha256sum -c --quiet
printf 'PASS: node archive raw size and SHA-256 match\n'
'''
    kube('bash', '-c', checker, '--', REMOTE_TAR, SOURCE_SHA256, str(SIZE), timeout=10800)


NODE_METADATA_CHECKER = r'''import hashlib
import json
import os
import re
import sys
import tarfile

path, raw_digest, expected_size, expected_bytes, target, config_digest = sys.argv[1:7]
check_raw = '--hash' in sys.argv[7:]
expected_size = int(expected_size)
expected_bytes = int(expected_bytes)
config_hex = config_digest.removeprefix('sha256:')
config_path = config_hex + '.json'
if os.stat(path).st_size != expected_size:
    raise SystemExit('Node archive raw size mismatch')
if check_raw:
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            h.update(chunk)
    if h.hexdigest() != raw_digest:
        raise SystemExit('Node archive raw SHA-256 mismatch')
    print('PASS: optional Python archive SHA-256 matches')


def stream(file):
    h = hashlib.sha256()
    total = 0
    while chunk := file.read(16 * 1024 * 1024):
        h.update(chunk)
        total += len(chunk)
    return h.hexdigest(), total


with tarfile.open(path, 'r:') as archive:
    # `r:` preserves the raw stream but cannot resume relative random access
    # after each streamed layer. `r` reads metadata once and safely seeks the
    # already verified node file, hashing every member exactly once.
    members = archive.getmembers()
    if any(member.islnk() or member.issym() or member.isdev() for member in members):
        raise SystemExit('Archive contains a link or device entry')
    files = {member.name: member for member in members if member.isfile()}
    if len(files) != sum(member.isfile() for member in members):
        raise SystemExit('Archive has duplicate file paths')
    manifest_member = files.get('manifest.json')
    if manifest_member is None or manifest_member.size > 1048576:
        raise SystemExit('Exactly one bounded Docker manifest.json is required')
    with archive.extractfile(manifest_member) as stream_file:
        descriptor = json.load(stream_file)
    if not isinstance(descriptor, list) or len(descriptor) != 1:
        raise SystemExit('Archive must contain exactly one image entry')
    entry = descriptor[0]
    if entry.get('Config') != config_path:
        raise SystemExit('Docker manifest names the wrong config member')
    if entry.get('RepoTags') != [target]:
        raise SystemExit('Archive does not contain exactly the requested target tag')
    layers = entry.get('Layers')
    if not isinstance(layers, list) or len(layers) != 75 or len(set(layers)) != 75:
        raise SystemExit('Archive must list 75 distinct layer member paths')
    for name in [config_path, *layers]:
        member = files.get(name)
        if (member is None or name.startswith('/') or '..' in name.split('/') or
                (name != config_path and not re.fullmatch(r'[0-9a-f]{64}/layer.tar', name))):
            raise SystemExit('Archive layer/config membership or safe path is invalid')
    if any(name.startswith('/') or '..' in name.split('/') for name in files):
        raise SystemExit('Archive contains an unsafe extra file path')
    config_members = [name for name in files if re.fullmatch(r'[0-9a-f]{64}\.json', name)]
    if config_members != [config_path]:
        raise SystemExit('Archive does not have exactly the expected single config member')
    allowed = {'manifest.json', config_path, 'repositories', *layers}
    if not set(files).issubset(allowed):
        raise SystemExit('Archive contains unexpected file members')
    with archive.extractfile(files[config_path]) as stream_file:
        config_raw = stream_file.read()
    if hashlib.sha256(config_raw).hexdigest() != config_hex:
        raise SystemExit('Image config bytes do not match expected config digest')
    config = json.loads(config_raw)
    diff_ids = (config.get('rootfs') or {}).get('diff_ids')
    if not isinstance(diff_ids, list) or len(diff_ids) != 75:
        raise SystemExit('Image config does not contain 75 rootfs diff_ids')
    total = len(config_raw)
    actual_ids = []
    for layer in layers:
        with archive.extractfile(files[layer]) as stream_file:
            layer_digest, count = stream(stream_file)
        if count != files[layer].size:
            raise SystemExit('Streamed layer size differs from its tar header')
        actual_ids.append('sha256:' + layer_digest)
        total += count
    if actual_ids != diff_ids:
        raise SystemExit('Ordered uncompressed layers differ from the source config')
    if total != expected_bytes:
        raise SystemExit('Config plus streamed layers total differs from expected bytes')
print('PASS: exact tag, one config, 75 layer paths/diff_ids; config+layers bytes:', total)
'''


def verify_node_archive_metadata(check_hash=False):
    """Read the Docker archive using host Python in chroot, never extraction.

    Raw hashing is optional here to avoid a second ~40 GB pass when
    verify_node_tar already used the present host sha256sum command.
    """
    argv = ['chroot', '/host', '/usr/bin/python3', '-', REMOTE_TAR.removeprefix('/host'),
            SOURCE_SHA256, str(SIZE), str(CONTENT_BYTES), TARGET, EXPECTED_CONFIG]
    if check_hash:
        argv.append('--hash')
    # The chroot Python program is stdin; registry and URL stdin are separate
    # invocations, so those protected streams are never mixed with source code.
    # Explicit concatenated chroot argv: `- "$@"` puts an extra `--` in
    # sys.argv before these six parameters, instead of forwarding them cleanly.
    command = 'set -euo pipefail; exec chroot /host /usr/bin/python3 - ' + shlex.join(argv[4:])
    kube('bash', '-c', command, input_text=NODE_METADATA_CHECKER, timeout=10800)


def download_to_node(alias):
    mc = os.environ.get('MC', 'mc')
    raw = subprocess.check_output([mc, '--insecure', 'share', 'download', '--expire', '2h',
                                   f'{alias}/oi-deployment-files/{PREFIX}/sglang-0.5.23-cu130-amd64.tar'], text=True)
    signed = [line.removeprefix('Share: ').strip() for line in raw.splitlines()
              if line.startswith('Share: ')]
    if len(signed) != 1 or not signed[0].startswith('https://') or 'X-Amz-Signature=' not in signed[0]:
        raise RuntimeError('MinIO must return exactly one HTTPS presigned archive URL')
    # Script comes from bash -c argv; the protected URL alone occupies stdin.
    cmd = r'''set -euo pipefail
IFS= read -r URL
A="$1"; EXPECTED="$2"; BYTES="$3"
command -v sha256sum >/dev/null
if [[ ! -f "$A" ]] || ! printf '%s  %s\n' "$EXPECTED" "$A" | sha256sum -c --quiet; then
  curl -fSk --retry 5 --retry-delay 10 --retry-all-errors --connect-timeout 30 --max-time 7200 -C - -o "$A" "$URL"
fi
printf '%s  %s\n' "$EXPECTED" "$A" | sha256sum -c --quiet
[[ "$(stat -c%s "$A")" == "$BYTES" ]]
printf 'PASS: node archive raw hash and size match\n'
'''
    kube('bash', '-c', cmd, '--', REMOTE_TAR, SOURCE_SHA256, str(SIZE),
         input_text=signed[0] + '\n', sensitive_input=True, timeout=10800)


def live_writer_exists():
    """Only PID output; never print cmdlines, which contain ctr Basic secrets."""
    cmd = r'''TARGET="$1"
for proc in /proc/[0-9]*; do
  [[ -r "$proc/comm" && -r "$proc/cmdline" ]] || continue
  [[ "$(cat "$proc/comm")" == ctr ]] || continue
  images=0; pushing=0; target=0
  while IFS= read -r -d '' arg; do
    [[ "$arg" == images ]] && images=1
    [[ "$arg" == push ]] && pushing=1
    [[ "$arg" == "$TARGET" ]] && target=1
  done < "$proc/cmdline"
  if ((images && pushing && target)); then
    printf '%s\n' "${proc##*/}"
  fi
done
'''
    return bool(kube('bash', '-c', cmd, '--', TARGET, timeout=60).stdout.strip())


def verify_imported_image():
    """Native list only: any absent/wrong tag or descriptor is a hard error."""
    cmd = r'''set -euo pipefail
T="$1"
CTR=(/host/usr/bin/ctr -a /host/run/containerd/containerd.sock -n k8s.io)
printf '%s\n' '---quiet---'
"${CTR[@]}" images list -q "name==$T"
printf '%s\n' '---descriptor---'
"${CTR[@]}" images list "name==$T"
'''
    result = kube('bash', '-c', cmd, '--', TARGET, timeout=60).stdout
    try:
        quiet, descriptor = result.split('---descriptor---\n', 1)
        quiet = quiet.removeprefix('---quiet---\n').splitlines()
        header, *rows = descriptor.strip().splitlines()
        fields = [row.split() for row in rows]
    except ValueError:
        raise RuntimeError('Unable to parse native ctr image-list output') from None
    if quiet != [TARGET] or not header.startswith('REF ') or len(fields) != 1:
        raise RuntimeError('Exact target is absent or ambiguous; retry will not import')
    if fields[0][0] != TARGET or len(fields[0]) < 3 or fields[0][2] != EXPECTED_MANIFEST:
        raise RuntimeError('Imported target manifest descriptor differs from the exact source')
    print('PASS: exact imported tag and expected native manifest descriptor', flush=True)


def import_image():
    if live_writer_exists():
        raise SystemExit('Exact-target ctr push is active; no import will be started')
    verify_node_archive_metadata()
    cmd = r'''set -euo pipefail
T="$1"
CTR=(/host/usr/bin/ctr -a /host/run/containerd/containerd.sock -n k8s.io)
"${CTR[@]}" images import /host/tmp/sglang-0.5.23-cu130-amd64.tar
[[ "$("${CTR[@]}" images list -q "name==$T")" == "$T" ]]
'''
    kube('bash', '-c', cmd, '--', TARGET, timeout=10800)
    verify_imported_image()


def push_only(creds, insecure_registry=False):
    credentials = creds[0] + ':' + creds[1]
    if live_writer_exists():
        raise SystemExit('Exact-target ctr publisher is active; inspect it instead of starting another')
    verify_imported_image()
    script = r'''set -euo pipefail
T="$1"; INSECURE="$2"
IFS= read -r REG_CREDS
[[ "$REG_CREDS" == *:* ]] || { echo 'Invalid protected Basic credential input'; exit 2; }
CTR=(/host/usr/bin/ctr -a /host/run/containerd/containerd.sock -n k8s.io)
FLAGS=()
if [[ "$INSECURE" == 1 ]]; then FLAGS+=(--skip-verify); fi
# ctr has no authfile option; its literal -u child argv remains a documented
# privileged-inspection limitation. exec leaves no second long-lived bash argv.
exec "${CTR[@]}" images push "${FLAGS[@]}" -u "$REG_CREDS" "$T"
'''
    result = kube('bash', '-c', script, '--', TARGET, '1' if insecure_registry else '0',
                  input_text=credentials + '\n', sensitive_input=True, timeout=14400)
    print(result.stdout, end='', flush=True)
    print('ctr push exited 0; exact registry readback required for delivery proof', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retry-push', action='store_true', help='Verify the existing exact import, push only, and read back; never download/import')
    parser.add_argument('--skip-download', action='store_true', help='Hash and inspect the existing node tar, then import it')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--discover-credentials-only', action='store_true', help='Resolve the existing 0600 exact-registry store and return before any network operation')
    parser.add_argument('--check-blobs', action='store_true', help='Also stream and hash the registry config and all 75 ordered compressed layer blobs')
    parser.add_argument('--tls-ca-bundle', help='CA bundle for Python registry reads; remote ctr uses its own node trust configuration')
    parser.add_argument('--insecure-registry', action='store_true', help='Explicitly disable both Python registry TLS verification and ctr certificate verification (--skip-verify); not HF/MinIO TLS')
    args = parser.parse_args()
    if args.retry_push and args.skip_download:
        parser.error('--retry-push already skips download; do not combine these flags')
    creds = existing_credentials()
    if args.discover_credentials_only:
        print('Exact-registry mode-0600 inline basic auth found; secret omitted', flush=True)
        return
    if args.insecure_registry and args.tls_ca_bundle:
        parser.error('Choose --tls-ca-bundle or --insecure-registry, not both')
    # None selects the system CA store. A private certificate error is not
    # silently bypassed: the operator must supply a CA or explicitly opt out.
    tls = ssl._create_unverified_context() if args.insecure_registry else ssl.create_default_context(cafile=args.tls_ca_bundle)
    if verify_registry(creds, tls, args.check_blobs):
        print('Delivery already verified; no transfer or deployment action needed', flush=True)
        return
    if args.verify_only:
        raise SystemExit(2)
    conditions = kube('get', '-n', 'pause-push', 'pod', 'img-staging', '-o',
                      'jsonpath={.spec.nodeName} {.status.phase} {.spec.hostPID} {.spec.containers[*].securityContext.privileged}', timeout=60).stdout.split()
    if (len(conditions) != 4 or conditions[:3] != ['adeo-gpu-02', 'Running', 'true'] or
            conditions[3] != 'true'):
        raise SystemExit('Staging must run on the target node with hostPID and privileged host-process access; no substitute workload pod')
    if live_writer_exists():
        raise SystemExit('Exact-target ctr publisher is currently active; no transfer or import will be started')
    if args.retry_push:
        verify_imported_image()
    else:
        alias = os.environ.get('MC_ALIAS', 'adeo-oicm-alain')
        check_minio(alias)
        if args.skip_download:
            verify_node_tar()
        else:
            download_to_node(alias)
        import_image()
    push_only(creds, args.insecure_registry)
    if not verify_registry(creds, tls, args.check_blobs):
        raise RuntimeError('No exact registry publication after push; delivery is not claimed')
    print('DELIVERY PROVEN: no model deployment changed', flush=True)


if __name__ == '__main__':
    main()
