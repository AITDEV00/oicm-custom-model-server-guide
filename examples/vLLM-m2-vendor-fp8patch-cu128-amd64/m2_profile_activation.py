"""Guarded profiler-only addition to one already-healthy tuned M2 Deployment.

User-authorized one controlled restart for a 32-token decode profile. No new
model/source code is applied. The profiler is ready only at worker init.
Only main metadata env entries are changed. Trace outputs stay in main /tmp;
result JSON/trace files are exported by a separate read-only transport.

No replay: current identity/replicas/resourceVersion/pod UID predetermined;
Generation 13->14 with Recreate. A separate tune restore script is invalid here,
because it guards zero replicas instead of healthy activation.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
import pathlib
import subprocess

DEPLOY = 'j-7ec325bf-0033-4a97-a688-e8ac30334349'
UID = 'cbe870c3-263f-4460-a6ae-05b9d4fb9b76'
TEMPLATE_HASH = '84fc97ff'
PROFILE_DIR = '/tmp/m2-decode-profile-1'
ENV = [
    {'name': 'VLLM_TORCH_PROFILER_DIR', 'value': PROFILE_DIR},
    {'name': 'VLLM_PROFILER_DELAY_ITERS', 'value': '0'},
    {'name': 'VLLM_PROFILER_MAX_ITERS', 'value': '32'},
    {'name': 'VLLM_TORCH_PROFILER_RECORD_SHAPES', 'value': '1'},
    {'name': 'VLLM_TORCH_PROFILER_WITH_STACK', 'value': '1'},
    {'name': 'VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY', 'value': '0'},
    {'name': 'VLLM_TORCH_PROFILER_DISABLE_ASYNC_LLM', 'value': '1'},
    {'name': 'VLLM_TORCH_PROFILER_USE_GZIP', 'value': '0'},
]


def relay(*args):
    cmd = ['kubectl', '-n', 'oik8s-cilium-system', 'exec', 'ad-debug', '--',
        'chroot', '/host', 'env', 'KUBECONFIG=/tmp/ad-kubeconfig', 'kubectl',
        '-n', 'adeo', *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def validate_current_tuned(dep: dict, pod: dict) -> None:
    assert dep['metadata']['uid'] == UID, dep['metadata']['uid']
    assert dep['metadata']['generation'] == 13
    assert dep['spec']['strategy']['type'] == 'Recreate'
    assert dep['spec']['replicas'] == dep['status']['readyReplicas'] == 1
    assert pod['metadata']['name'].startswith(DEPLOY+'-'+TEMPLATE_HASH+'-')
    assert pod['status']['phase'] == 'Running' and pod['spec']['nodeName'] == 'prd-infr-k8h200'
    cs = next(x for x in pod['status']['containerStatuses'] if x['name'] == 'main')
    assert cs['ready'] and cs['restartCount'] == 0
    main = next(c for c in dep['spec']['template']['spec']['containers'] if c['name'] == 'main')
    assert any(v == '/etc/m2-moe-tuned' for v in [next(e['value'] for e in main['env'] if e['name'] == 'VLLM_TUNED_CONFIG_FOLDER')])
    for image in main['image'], next(c['image'] for c in pod['spec']['containers'] if c['name'] == 'main'):
        assert 'm2-vllm@sha256:d3ced94ef036e8469bbbb85b8973a31c9b48bf441b565db3b3197cef36309ba9' in image
    assert not any(e['name'].startswith('VLLM_PROFILER_') or e['name'] in (
        'VLLM_TORCH_PROFILER_DIR', 'VLLM_TORCH_CUDA_PROFILE') for e in main['env'])


def make_guarded_plan(before: dict, before_pod: dict) -> dict:
    validate_current_tuned(before, before_pod)
    desired = copy.deepcopy(before)
    desired['metadata']['resourceVersion'] = None
    main = next(c for c in desired['spec']['template']['spec']['containers'] if c['name'] == 'main')
    main['env'] += ENV
    # Deployment PATCH delta legitimately includes replicas, metadata RV.
    patch = {'spec': {
        'replicas': 1,
        'template': desired['spec']['template'],
    }}
    expected = copy.deepcopy(before['spec']['template'])
    assert patch['spec']['template'] != expected
    expected_main = next(c for c in expected['spec']['containers'] if c['name'] == 'main')
    patch_main = next(c for c in patch['spec']['template']['spec']['containers'] if c['name'] == 'main')
    assert patch_main['env'][:-len(ENV)] == expected_main['env']
    assert patch_main['env'][-len(ENV):] == ENV
    for changed, before_c in zip(patch['spec']['template']['spec']['containers'], expected['spec']['containers']):
        x = copy.deepcopy(changed); y = copy.deepcopy(before_c)
        x.pop('env', None); y.pop('env', None)
        assert x == y
    # With main env normalized back, the podTemplate itself must be byte-equal.
    normalized = copy.deepcopy(patch['spec']['template'])
    nm = next(c for c in normalized['spec']['containers'] if c['name'] == 'main')
    nm['env'] = expected_main['env']
    assert normalized == expected
    return {'expected_current_uid': UID,
            'expected_current_resourceVersion': before['metadata']['resourceVersion'],
            'expected_current_generation': 13,
            'probe_pod_uid': before_pod['metadata']['uid'],
            'probe_pod_main_id': next(s['containerID'] for s in before_pod['status']['containerStatuses'] if s['name'] == 'main'),
            'allowed_delta': 'ONLY eight profile envs on main; no code/weights/parallelism changes',
            'profile_capture': 'One warm32token single-stream, trace max32 scheduler iterations',
            'deployment_patch': patch}


def apply(plan: dict):
    fresh = json.loads(relay('get', 'deployment', DEPLOY, '-o', 'json'))
    assert fresh['metadata']['uid'] == plan['expected_current_uid']
    assert fresh['metadata']['resourceVersion'] == plan['expected_current_resourceVersion']
    assert fresh['metadata']['generation'] == plan['expected_current_generation']
    assert fresh['status']['readyReplicas'] == 1
    fresh_pods = json.loads(relay('get', 'pods', '-o', 'json'))['items']
    selected = [p for p in fresh_pods if p['metadata']['uid'] == plan['probe_pod_uid']]
    assert len(selected) == 1
    c = next(s for s in selected[0]['status']['containerStatuses'] if s['name'] == 'main')
    assert c['ready'] and c['containerID'] == plan['probe_pod_main_id'] and c['restartCount'] == 0
    relay('patch', 'deployment', DEPLOY, '--type=strategic', '-p',
          json.dumps(plan['deployment_patch'], separators=(',', ':')))
    return {'profiler_activation_sent': True, 'from_generation': 13,
            'expected_new_generation': 14, 'source_changed': False}


def main():
    os.environ.setdefault('KUBECONFIG', str(pathlib.Path.home()/'.kube/alain-oicm.conf'))
    before = json.loads(relay('get', 'deployment', DEPLOY, '-o', 'json'))
    pod = json.loads(relay('get', 'pods', '-l', 'io.kompose.service='+DEPLOY, '-o', 'json'))['items']
    assert len(pod) == 1
    plan = make_guarded_plan(before, pod[0])
    dest = pathlib.Path(__file__).parent/'h200-moe-measurements/profile-activation-plan.json'
    dest.write_text(json.dumps(plan, indent=2)+'\n')
    snap = pathlib.Path(__file__).parent/'h200-moe-measurements/pre-profile-tuned-gen13.json'
    snap.write_text(json.dumps(before, indent=2)+'\n')
    print('EXACT_GUARDS_PASSED_AND_PLAN_SAVED_NOT_YET_APPLIED')
    print(json.dumps(apply(plan)))
    print('NEXT WAIT readiness, warm_once_unchanged_request_then_one32token_capture')


if __name__ == '__main__':
    main()
