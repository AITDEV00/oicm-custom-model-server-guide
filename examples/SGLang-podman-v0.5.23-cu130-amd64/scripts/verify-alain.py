#!/usr/bin/env python3
"""Verify exact Al Ain Harbor identity without exposing registry credentials.

Credentials come from an approved existing container-auth JSON store or a
separately supplied, protected operator file. Never print credentials or URLs.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import ssl
import urllib.error
import urllib.request

REGISTRY = 'registry.adeoaiengine.ecouncil.ae'
REPO = 'openinnovationai/platform/mlops/mlops-serving/sglang'
REF = f'{REGISTRY}/{REPO}:0.5.23-cu130-amd64'
CONFIG = 'sha256:ab8fc368a60473e329bde89157d84b20ce3bef8fddd5b6bf4999df65c4392cf5'
MANIFEST = 'sha256:01e0d5637028f59e2836bfc8c385c393a9d722d4b971acbcfa66875569eee960'


def credentials(store: str | None) -> tuple[str, str]:
    """Resolve the user's existing store without writing decoded values."""
    named = Path(store).expanduser() if store else None
    if named and not named.suffix == '.json':
        raw = named.read_text().strip()
        user, sep, secret = raw.partition(':')
        assert sep and user and secret, 'Protected file must be user:password'
        return user, secret
    paths = [named] if named else [
        Path('/run/user') / str(os.getuid()) / 'containers/auth.json',
        Path.home() / '.docker/config.json',
        Path.home() / '.config/containers/auth.json',
    ]
    for candidate in paths:
        if candidate and candidate.is_file():
            content = candidate.read_text()
            if content.startswith('authfile:'):
                return credentials(content.removeprefix('authfile:').strip())
            doc = json.loads(content)
            if doc.get('credStore') or doc.get('credsStore'):
                raise SystemExit('Protected credential helper configured: use its standard registry tooling rather than decoding its store.')
            entry = (doc.get('auths') or {}).get(REGISTRY)
            if isinstance(entry, dict) and entry.get('auth'):
                raw = base64.b64decode(entry['auth']).decode()
                user, sep, secret = raw.partition(':')
                if sep:
                    return user, secret
    raise SystemExit('Protected registry store not configured; reuse standard container auth or an authorized existing protected operator file.')


def query(ref: str, path: str, headers: dict, ctx: ssl.SSLContext):
    host, repo_tag = ref.split('/', 1)
    repo, tag = repo_tag.rsplit(':', 1)
    url = f'https://{host}/v2/{repo}/{path.format(tag=tag)}'
    request = urllib.request.Request(url, headers=headers)
    try:
        return urllib.request.urlopen(request, context=ctx, timeout=90)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'Registry GET returned HTTP {exc.code}; response body suppressed') from None


def verify(store: str | None, check_blobs: bool, tls_ca: str | None, insecure: bool = False) -> None:
    user, secret = credentials(store)
    authorization = base64.b64encode(f'{user}:{secret}'.encode()).decode()
    headers = {
        'Authorization': 'Basic ' + authorization,
        'Accept': 'application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json',
    }
    ctx = ssl._create_unverified_context() if insecure else ssl.create_default_context(cafile=tls_ca)
    with query(REF, 'manifests/{tag}', headers, ctx) as response:
        assert response.status == 200
        body = response.read()
        actual_manifest = response.headers.get('Docker-Content-Digest')
    try:
        manifest = json.loads(body)
    except ValueError:
        raise SystemExit('Registry manifest is not JSON') from None
    assert body and hashlib.sha256(body).hexdigest() == MANIFEST.split(':')[1]
    assert actual_manifest == MANIFEST, (actual_manifest, MANIFEST)
    assert (manifest.get('config') or {}).get('digest') == CONFIG
    assert manifest.get('mediaType') == 'application/vnd.docker.distribution.manifest.v2+json'
    assert len(manifest.get('layers', [])) == 75
    print('PASS: Harbor HTTP 200; exact reference', REF, flush=True)
    print('PASS: image config', CONFIG, flush=True)
    print('PASS: registry manifest', MANIFEST, flush=True)
    print('PASS: 75 ordered layer descriptors', flush=True)
    if check_blobs:
        for item in [manifest['config'], *manifest['layers']]:
            digest = item['digest']
            with query(REF, 'blobs/' + digest, headers, ctx) as response:
                h = hashlib.sha256()
                count = 0
                while chunk := response.read(16 * 1024 * 1024):
                    h.update(chunk)
                    count += len(chunk)
            assert count == item['size'], (count, item['size'])
            assert h.hexdigest() == digest.split(':')[1]
        print('PASS: registry served the config and all 75 blobs with matching size/SHA-256', flush=True)
    print('RUNTIME IMAGES ONLY: no deployment was changed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--auth-store', help='Standard auth.json, authfile:<file>, or legacy protected user:password file; no credentials in arguments')
    parser.add_argument('--check-blobs', action='store_true', help='Additional ~37 GB readback to hash config/all layer blobs (optional, not required for proven manifest vs source)')
    parser.add_argument('--tls-ca-bundle', help='Existing trusted CA bundle, if standard validation is available')
    parser.add_argument('--insecure-registry', action='store_true', help='Scoped compatibility for the existing Al Ain Harbor certificate only; never applies to HF')
    args = parser.parse_args()
    if args.tls_ca_bundle and args.insecure_registry:
        parser.error('Choose --tls-ca-bundle or --insecure-registry, not both')
    verify(args.auth_store, args.check_blobs, args.tls_ca_bundle, args.insecure_registry)
