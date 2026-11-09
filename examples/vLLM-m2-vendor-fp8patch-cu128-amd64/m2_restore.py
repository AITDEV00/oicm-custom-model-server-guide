#!/usr/bin/env python3
"""Static plan for the exact pre-maintenance M2 Deployment template.

Does not re-download weights, touch model/PVC or license. If a native tuned JSON
is supplied, only add its ConfigMap path and environment to the original spec.
The caller applies to the exact UID after independent native A/B/loading gates.
Untuned restore has no Triton/dtype or checkpoint dependence.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

UID = 'cbe870c3-263f-4460-a6ae-05b9d4fb9b76'
DEPLOY = 'j-7ec325bf-0033-4a97-a688-e8ac30334349'
PVC = 'pvc-7ec325bf-0033-4a97-a688-e8ac30334349'
DIGEST = 'sha256:d3ced94ef036e8469bbbb85b8973a31c9b48bf441b565db3b3197cef36309ba9'
ARTIFACT = 'E=24,N=1792,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json'
CM = 'm2-fp8-moe-tuned-config'
FOLDER = '/etc/m2-moe-tuned'
KEYS = {'BLOCK_SIZE_M': {16, 32, 64, 128, 256}, 'BLOCK_SIZE_N': {128, 256},
        'BLOCK_SIZE_K': {128}, 'GROUP_SIZE_M': {1, 16, 32, 64},
        'num_warps': {4, 8}, 'num_stages': {2, 3, 4, 5}}


def main(original_path, tuned_path, current_path=None):
    orig = json.loads(Path(original_path).read_text())
    assert orig['metadata']['uid'] == UID and orig['metadata']['name'] == DEPLOY
    assert orig['metadata']['generation'] == 11
    assert orig['spec']['replicas'] == 1 and orig['spec']['strategy']['type'] == 'Recreate'
    template = copy.deepcopy(orig['spec']['template'])
    main = next(c for c in template['spec']['containers'] if c['name'] == 'main')
    assert main['image'].endswith('@' + DIGEST)
    assert main['resources']['limits']['nvidia.com/h200'] == '8'
    assert main['args'][0] == '/pvc-home/app/download/base_model'
    for k, v in {'--max-model-len': '262144', '--served-model-name': 'M2',
                 '--tensor-parallel-size': '8', '--kv-cache-dtype': 'fp8_e4m3'}.items():
        assert main['args'][main['args'].index(k) + 1] == v
    assert '--enable-expert-parallel' in main['args']
    assert main['args'][main['args'].index('--override-generation-config') + 1] == '{"eos_token_id":250019}'
    assert not main.get('command'), 'stock image entrypoint is required'
    next(c for c in template['spec']['initContainers'] if c['name'] == 'init-model-downloader')
    vol = next(v for v in template['spec']['volumes'] if v['name'].startswith('pvc-'))
    assert vol['persistentVolumeClaim']['claimName'] == PVC
    assert any(m['mountPath'] == '/pvc-home' and not m.get('subPath') for m in main['volumeMounts'])
    assert next(m for m in main['volumeMounts'] if m['mountPath'] == '/license')['readOnly']
    assert not any(m.get('mountPath', '').endswith('/xllm.py') for m in main['volumeMounts'])
    create = None
    if tuned_path:
        path = Path(tuned_path)
        native = json.loads(path.read_text())
        assert native['triton_version'] == '3.5.0'
        assert set(native) == {'triton_version', '1', '2', '4', '8'}
        for m in ('1', '2', '4', '8'):
            assert set(native[m]) == set(KEYS)
            assert all(native[m][k] in allowed for k, allowed in KEYS.items())
        ab = json.loads((path.parent / 'finalist-ab.json').read_text())
        assert set(ab) == {'1', '2', '4', '8'}
        for m, row in ab.items():
            assert row['numerical_pass'] and row['speedup_local_expert'] > 1.0
            assert row['candidate'] == native[m]
        loader = json.loads((path.parent / 'native-loader-verification.json').read_text())
        assert loader['validated_vendor_loader'] is True
        assert loader['production_mutation_count'] == 0
        assert hashlib.sha256(path.read_bytes()).hexdigest() == loader['sha256']
        assert all(e['name'] != 'VLLM_TUNED_CONFIG_FOLDER' for e in main.get('env', []))
        assert all(v['name'] != CM for v in template['spec']['volumes'])
        main.setdefault('env', []).append({'name': 'VLLM_TUNED_CONFIG_FOLDER', 'value': FOLDER})
        main['volumeMounts'].append({'name': 'm2-moe-tuned-config', 'mountPath': FOLDER, 'readOnly': True})
        template['spec']['volumes'].append({'name': 'm2-moe-tuned-config', 'configMap': {
            'name': CM, 'items': [{'key': 'native.json', 'path': ARTIFACT}]}})
        create = {'apiVersion': 'v1', 'kind': 'ConfigMap', 'metadata': {'name': CM,
                   'namespace': 'adeo'}, 'data': {'native.json':
                   json.dumps(native, sort_keys=True, indent=2)}}
    if current_path:
        current = json.loads(Path(current_path).read_text())
        assert current['metadata']['uid'] == UID
        assert current['spec']['replicas'] == 0
        assert current['spec']['template']['spec']['runtimeClassName'] == 'nvidia'
        assert not current.get('status', {}).get('replicas', 0)
    return {'deployment_patch': {'spec': {'replicas': 1, 'template': template}},
            'artifact_configmap_create': create,
            'preserves_model_pvc_downloader_license': True,
            'source_mount_change': bool(tuned_path),
            'stock_image_digest': DIGEST}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('original')
    p.add_argument('--tuned-config')
    p.add_argument('--current')
    args = p.parse_args()
    print(json.dumps(main(args.original, args.tuned, args.current), indent=2))
