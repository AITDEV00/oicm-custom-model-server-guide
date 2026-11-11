#!/usr/bin/env python3
"""Export the existing SGLang image to a durable Docker archive for HF.

For a NEW first-time publication only. This wrapper does not validate or start
Al Ain MinIO ingestion, does not push Harbor, and does not deploy a model.
It refuses to republish an already-proven archive. Read HF_RELAY.md for the
handoff, then reuse the pinned hfs3 config instead of moving HF revision main.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile

DIR = Path(__file__).resolve().parent
SOURCE = 'localhost/oicm/sglang:0.5.23-cu130-amd64'
TARGET = 'registry.adeoaiengine.ecouncil.ae/openinnovationai/platform/mlops/mlops-serving/sglang:0.5.23-cu130-amd64'
EXPECTED_CONFIG = 'ab8fc368a60473e329bde89157d84b20ce3bef8fddd5b6bf4999df65c4392cf5'
EXPECTED_SHA256 = '570648111409bdece6ec93192643808a30787311059c90f451e487cfe145c20a'
FULL_SIZE = 39_787_218_432
DIR_NAME = 'image-archive'
NAME = 'sglang-0.5.23-cu130-amd64.tar'


def tool(*args: str):
    result = subprocess.run(args, text=True, capture_output=True, check=True)
    return result.stdout


def validate(archive: Path) -> None:
    assert archive.is_file(), f'Missing durable Docker archive: {archive}'
    assert archive.stat().st_size == FULL_SIZE
    with archive.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == EXPECTED_SHA256
    with tarfile.open(archive, 'r:') as packaged:
        records = json.load(packaged.extractfile('manifest.json'))
        assert len(records) == 1
        assert records[0]['Config'] == EXPECTED_CONFIG + '.json'
        assert records[0]['RepoTags'] == [TARGET]
        assert len(records[0]['Layers']) == 75
        names = {member.name for member in packaged.getmembers()}
        assert set([records[0]['Config'], *records[0]['Layers']]).issubset(names)
        config = packaged.extractfile(records[0]['Config']).read()
        assert hashlib.sha256(config).hexdigest() == EXPECTED_CONFIG
        total = len(config)
        for layer in records[0]['Layers']:
            with packaged.extractfile(layer) as stream:
                while chunk := stream.read(16 * 1024 * 1024):
                    total += len(chunk)
        assert total == 39_786_978_626
    print('PASS: exact archive hash/size, tag, config, 75 layers/aggregate bytes')


def main() -> None:
    archive = DIR / DIR_NAME / NAME
    source_id = tool('podman', 'image', 'inspect', '--format', '{{.Id}}', SOURCE).strip().removeprefix('sha256:')
    if source_id != EXPECTED_CONFIG:
        raise SystemExit(f'Current image config {source_id} differs from proven artifact; treat as NEW artifact with new identity, revision, credentials scope and documentation')
    if archive.exists():
        validate(archive)
        raise SystemExit('Exact archive already exists; refusing to re-export/republish. Existing Harbor identity is recorded in HF_RELAY.md.')
    if os.environ.get('INITIATE_PUBLIC_HF_RELAY') != 'I_HAVE_APPROVED_PUBLIC_ARTIFACT_TRANSIT':
        raise SystemExit('Set INITIATE_PUBLIC_HF_RELAY with this exact attestation only after public-transit/licensing/disclosure approval. No HF upload occurs otherwise; no manual credentials are required.')
    (DIR / DIR_NAME).mkdir(exist_ok=True)
    project_dir = DIR / DIR_NAME / '.hf-public-upload'
    project_dir.mkdir(exist_ok=True)
    print('Exporting the known existing source image in Docker format to the persistent worktree', flush=True)
    subprocess.run(['podman', 'tag', SOURCE, TARGET], check=True)
    subprocess.run(['podman', 'save', '--format=docker-archive', '-o', str(archive), TARGET], check=True)
    validate(archive)
    print('Local archive protected. Writing public metadata only via HF uploader after committed LFS SHA verification.', flush=True)
    subprocess.run([sys.executable, str(DIR / 'upload-public-hf.py')], check=True)
    print('HF tar committed and verified. Run bastion hfs3 with pinned config; do not assume Harbor contains tag yet.', flush=True)


if __name__ == '__main__':
    main()
