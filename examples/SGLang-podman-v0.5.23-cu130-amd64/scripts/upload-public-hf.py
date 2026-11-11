"""Publish the verified Docker archive using HF's resumable large-file workflow."""
from pathlib import Path
import hashlib
import json
import tarfile

from huggingface_hub import HfApi

REPO = 'maxJasper/sglang-0.5.23-cu130-amd64'
ARCHIVE = Path(__file__).resolve().parent.parent / 'image-archive' / 'sglang-0.5.23-cu130-amd64.tar'
assert ARCHIVE.is_file(), f'Missing export: {ARCHIVE}'
# Explicit authentication overrides the inherited read-only HF_TOKEN from the
# shell that opened the masked write login. Never print or persist this value.
token = Path.home().joinpath('.cache/huggingface/token').read_text().strip()
api = HfApi(token=token)
who = api.whoami()
assert who['name'].casefold() == 'maxjasper', f'Expected maxJasper, got {who["name"]}'
permissions = who.get('auth', {}).get('accessToken', {}).get('role')
assert permissions in {'write', 'admin'}, f'HF write permission required; role is {permissions}'
info = api.repo_info(REPO, repo_type='model')
assert not info.private, 'Target must be public for bastion HF access without credentials'

with tarfile.open(ARCHIVE) as archive:
    manifests = json.loads(archive.extractfile('manifest.json').read())
    names = {member.name for member in archive.getmembers()}
    required = {
        layer
        for image in manifests
        for layer in [image['Config'], *image['Layers']]
    }
    missing = sorted(required - names)
    assert not missing, f'Docker archive is missing required image files: {missing[:3]}'
    # Docker archives also contain directory metadata and legacy per-layer JSON;
    # those are allowed extras, unlike absent manifest-referenced files.
    tags = {tag for image in manifests for tag in (image.get('RepoTags') or [])}
    assert 'registry.adeoaiengine.ecouncil.ae/openinnovationai/platform/mlops/mlops-serving/sglang:0.5.23-cu130-amd64' in tags, tags
print('Verified complete Docker archive and intended Harbor tag.', flush=True)
print('Computing source archive SHA-256.', flush=True)
with ARCHIVE.open('rb') as source:
    digest = hashlib.file_digest(source, 'sha256').hexdigest()
print(f'Archive size: {ARCHIVE.stat().st_size} bytes; source SHA-256: {digest}', flush=True)
print('Publishing with HF resumable large-folder workflow.', flush=True)
api.upload_large_folder(
    repo_id=REPO,
    repo_type='model',
    folder_path=ARCHIVE.parent,
    private=False,
    allow_patterns=[ARCHIVE.name],
    num_workers=1,
    print_report_every=30,
)
tree=api.list_repo_tree(REPO,repo_type='model',recursive=True,expand=True)
files={x.path:x for x in tree if getattr(x,'type',None)=='file'}
archive = files[ARCHIVE.name]
assert archive.size==ARCHIVE.stat().st_size, 'HF returned wrong archive size'
assert archive.lfs is not None, 'Committed HF archive must use LFS'
assert archive.lfs.sha256==digest, 'Committed HF LFS SHA-256 differs from source archive'
print(f'Archive size: {archive.size} bytes; verified HF LFS SHA-256: {digest}', flush=True)
readme = f'''# SGLang runtime image: v0.5.23 CUDA 13.0 OICM

This public repository holds a **Docker runtime image archive**, not model weights.

- Runtime distribution: SGLang 0.5.23
- OCI image config: `sha256:ab8fc368a60473e329bde89157d84b20ce3bef8fddd5b6bf4999df65c4392cf5`
- Archive: `{ARCHIVE.name}`
- File size: {archive.size} bytes
- Archive SHA-256: `{digest}`
- Intended Al Ain Harbor repository:
  `registry.adeoaiengine.ecouncil.ae/openinnovationai/platform/mlops/mlops-serving/sglang:0.5.23-cu130-amd64`

The Docker archive embeds the intended target tag; retain it on import. The
source image config digest above must match Harbor before use.

Public artifact only. Do not place credentials, keys, models, or customer data
in this repository. Uploaded as a controlled transport relay; image provenance
is the local runtime build, not an upstream SGLang artifact signing guarantee.
'''
print('Publishing integrity metadata and README matching the verified LFS digest.', flush=True)
api.upload_file(
    path_or_fileobj=f'{digest}  {ARCHIVE.name}\n'.encode(),
    path_in_repo='manifest.sha256',
    repo_id=REPO,
    repo_type='model',
    commit_message='Record verified Docker archive SHA-256',
)
api.upload_file(
    path_or_fileobj=json.dumps({'archive': ARCHIVE.name, 'sha256': digest, 'size': archive.size}).encode() + b'\n',
    path_in_repo='sglang-0.5.23-cu130-amd64.transfer.json',
    repo_id=REPO,
    repo_type='model',
    commit_message='Record verified Docker archive transfer metadata',
)
api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo='README.md',
    repo_id=REPO,
    repo_type='model',
    commit_message='Describe public Docker runtime image transport relay',
)
print('Verified public HF archive size and committed LFS SHA-256.',flush=True)
print('Public URL:', f'https://huggingface.co/{REPO}/resolve/main/{ARCHIVE.name}',flush=True)
